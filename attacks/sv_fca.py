"""SV-FCA: Source-View Frequency-Coordinated Attack.

Final frequency-centric transfer attack used by Tables I-VIII.

Pipeline per attack step
------------------------
1. Build a source-view gradient pool over K source models x R differentiable views.
2. L1-normalize every source-view gradient before aggregation.
3. Decompose every normalized gradient into B radial rFFT bands.
4. Stream per-band mean gradients and exact average pairwise cosine consensus.
5. Coordinate bands using consensus + a low/mid transfer prior.
6. Stabilize per-image band weights with spectral EMA memory.
7. Reconstruct the frequency-coordinated direction, add ordinary attack momentum,
   and perform the projected L_inf update.

Table-V ablations
-----------------
- full_model
- without_sv_pool
- without_frequency_coordination
- without_band_consensus
- without_low_mid_prior
- without_spectral_memory

No token branch, target-model query, target gradient, target logits, or target
confidence is used during adversarial-example generation.
"""
from contextlib import nullcontext
import math
import torch

from attacks.base import Attack, AttackResult, input_diversity, loss_ce, normalize_grad_l1, pgd_update

_EPS = 1e-12


def _autocast_context(device, enabled=False, dtype_name="fp16"):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if str(dtype_name).lower() == "bf16" else torch.float16
    try:
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)


def _make_view(x_req, view_id, num_views, diversity_prob, image_size, resize_size):
    # View 0 is identity. Other views use the project's differentiable DI transform.
    if view_id == 0 or num_views <= 1:
        return x_req
    return input_diversity(
        x_req,
        prob=diversity_prob,
        image_size=image_size,
        resize_size=resize_size,
    )


def _radial_rfft_masks(height, width, num_bands, device):
    if num_bands < 2:
        raise ValueError("SV-FCA requires spectral_bands >= 2")
    fy = torch.fft.fftfreq(height, d=1.0, device=device)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(fy.square() + fx.square())
    max_radius = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    radius = (radius / max_radius).clamp(0.0, 1.0)
    edges = torch.linspace(0.0, 1.0, steps=num_bands + 1, device=device)
    masks = []
    for b in range(num_bands):
        if b == num_bands - 1:
            mask = ((radius >= edges[b]) & (radius <= edges[b + 1])).float()
        else:
            mask = ((radius >= edges[b]) & (radius < edges[b + 1])).float()
        masks.append(mask.view(1, 1, height, width // 2 + 1))
    return masks


def _l2_unit(x):
    flat = x.flatten(1)
    denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(_EPS)
    return x / denom.view(-1, 1, 1, 1)


def _pairwise_consensus_from_unit_sum(unit_sum, n, squared_norm_sum=None):
    """Exact mean pairwise cosine from the sum of unit vectors, per image.

    For u_i = g_i / max(||g_i||_2, eps), including zero gradients,
      ||sum_i u_i||^2 = sum_i ||u_i||^2 + 2 * sum_{i<j} <u_i,u_j>.
    The diagonal sum is n only when all normalized vectors have unit norm.
    n=1 uses the neutral consensus convention 1 (no pair to compare).
    """
    if n <= 1:
        return torch.ones(unit_sum.shape[0], device=unit_sum.device, dtype=torch.float32)
    sq = unit_sum.flatten(1).pow(2).sum(dim=1)
    diagonal = float(n) if squared_norm_sum is None else squared_norm_sum
    consensus = (sq - diagonal) / float(n * (n - 1))
    return consensus.clamp(-1.0, 1.0)


def _low_mid_prior(num_bands, device, strength=1.0, floor=0.05, center=0.30, width=0.22):
    """Return a normalized low/mid transfer prior for each radial band."""
    if num_bands < 2:
        raise ValueError("num_bands must be at least 2")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("floor must be in (0, 1]")
    if not math.isfinite(center) or not 0 <= center <= 1:
        raise ValueError("center must be in [0, 1]")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be positive")
    centers = (torch.arange(num_bands, device=device, dtype=torch.float32) + 0.5) / float(num_bands)
    # A focused, positive peak avoids the nearly uniform weights of the old prior.
    base = 0.10 + torch.exp(-0.5 * ((centers - center) / width).pow(2))
    base = base / base.max().clamp_min(_EPS)
    return base.clamp_min(floor).pow(strength)


def _band_energy_features(band_energy, eps=_EPS):
    """Convert source-view band energies to centered log-energy features.

    The feature is zero when every band has equal energy and positive for a
    band carrying more energy than the per-image mean.  Centering prevents the
    absolute gradient scale from changing the softmax temperature.
    """
    if band_energy.ndim != 3:
        raise ValueError("band_energy must have shape [batch, bands, 1] or [batch, bands, channels]")
    energy = band_energy.float().mean(dim=-1)
    reference = energy.mean(dim=1, keepdim=True).clamp_min(eps)
    return torch.log(energy.clamp_min(eps)) - torch.log(reference)


def _compute_band_weights(
    consensus,
    band_energy,
    band_temperature,
    low_mid_strength,
    consensus_gain=2.0,
    energy_strength=0.75,
    weight_floor=0.02,
    use_consensus=True,
    use_prior=True,
):
    """Fuse consensus, relative energy and prior into per-image band weights."""
    if consensus.ndim != 2:
        raise ValueError("consensus must have shape [batch, bands]")
    if band_energy.ndim != 2 or band_energy.shape != consensus.shape:
        raise ValueError("band_energy must have shape [batch, bands]")
    temperature = float(band_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("band_temperature must be finite and positive")
    if not math.isfinite(consensus_gain) or consensus_gain < 0:
        raise ValueError("consensus_gain must be finite and nonnegative")
    if not math.isfinite(energy_strength) or energy_strength < 0:
        raise ValueError("energy_strength must be finite and nonnegative")
    if not math.isfinite(weight_floor) or not 0 <= weight_floor <= 1.0 / consensus.shape[1]:
        raise ValueError("weight_floor must be in [0, 1 / num_bands]")

    logits = torch.zeros_like(consensus, dtype=torch.float32)
    if use_consensus:
        logits = logits + float(consensus_gain) * consensus.float()
    if energy_strength > 0:
        energy_features = _band_energy_features(band_energy.unsqueeze(-1))
        logits = logits + float(energy_strength) * energy_features
    if use_prior:
        prior = _low_mid_prior(consensus.shape[1], consensus.device, low_mid_strength)
        logits = logits + torch.log(prior.clamp_min(_EPS)).view(1, -1)
    weights = torch.softmax(logits / temperature, dim=1)
    # Keep every band weakly represented so one noisy FFT bin cannot erase all
    # other transfer directions; the vector is renormalized per image.
    if weight_floor > 0:
        weights = weights.clamp_min(float(weight_floor))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(_EPS)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("SV-FCA produced nonfinite spectral band weights")
    return weights


def _variant_flags(raw_variant, default_views):
    v = str(raw_variant).strip().lower()
    aliases = {
        "full": "full_model",
        "ours": "full_model",
        "sv_fca": "full_model",
        "svfca": "full_model",
        "without_source_view_pool": "without_sv_pool",
        "without_frequency": "without_frequency_coordination",
        "without_frequency_branch": "without_frequency_coordination",
        "without_consensus": "without_band_consensus",
        "without_prior": "without_low_mid_prior",
        "without_memory": "without_spectral_memory",
    }
    v = aliases.get(v, v)
    allowed = {
        "full_model",
        "without_sv_pool",
        "without_frequency_coordination",
        "without_band_consensus",
        "without_low_mid_prior",
        "without_spectral_memory",
    }
    if v not in allowed:
        raise ValueError("Unknown SV-FCA variant '{}'. Allowed: {}".format(raw_variant, sorted(allowed)))
    return {
        "variant": v,
        "num_views": 1 if v == "without_sv_pool" else max(1, int(default_views)),
        "frequency_coordination": v != "without_frequency_coordination",
        "band_consensus": v != "without_band_consensus",
        "low_mid_prior": v != "without_low_mid_prior",
        "spectral_memory": v != "without_spectral_memory",
    }


def _stream_source_view_frequency_stats(
    models,
    x,
    y,
    num_views,
    spectral_bands,
    diversity_prob,
    image_size,
    resize_size,
    need_frequency=True,
    need_consensus=True,
    amp=False,
    amp_dtype="fp16",
):
    """Stream source/view gradients; do not keep KxR gradients in GPU memory."""
    if len(models) < 1:
        raise ValueError("SV-FCA requires at least one source model")

    spatial_sum = torch.zeros_like(x, dtype=torch.float32)
    band_sums = None
    unit_sums = None
    unit_squared_norm_sums = None
    band_energy_sums = None
    masks = None
    pool_count = 0
    loss_sum = 0.0
    source_names = []

    for source_name, model in models.items():
        source_names.append(source_name)
        model.eval()
        for view_id in range(max(1, int(num_views))):
            x_req = x.detach().clone().requires_grad_(True)
            x_in = _make_view(x_req, view_id, num_views, diversity_prob, image_size, resize_size)
            with _autocast_context(x.device, enabled=amp, dtype_name=amp_dtype):
                logits = model(x_in)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = loss_ce(logits, y)
            grad = torch.autograd.grad(loss, x_req, retain_graph=False, create_graph=False)[0]
            grad_n = normalize_grad_l1(grad.detach().float())
            if not torch.isfinite(grad_n).all():
                raise FloatingPointError(
                    "SV-FCA received nonfinite input gradients. Disable AMP or use bf16, "
                    "and check the source model/input."
                )
            spatial_sum.add_(grad_n)

            if need_frequency:
                h, w = grad_n.shape[-2:]
                if masks is None:
                    masks = _radial_rfft_masks(h, w, spectral_bands, grad_n.device)
                    band_sums = [torch.zeros_like(grad_n, dtype=torch.float32) for _ in range(spectral_bands)]
                    band_energy_sums = [torch.zeros(x.shape[0], device=x.device, dtype=torch.float32) for _ in range(spectral_bands)]
                    if need_consensus:
                        unit_sums = [torch.zeros_like(grad_n, dtype=torch.float32) for _ in range(spectral_bands)]
                        unit_squared_norm_sums = [
                            torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
                            for _ in range(spectral_bands)
                        ]
                spectrum = torch.fft.rfft2(grad_n, dim=(-2, -1), norm="ortho")
                for b, mask in enumerate(masks):
                    band = torch.fft.irfft2(spectrum * mask, s=(h, w), dim=(-2, -1), norm="ortho").real
                    band_sums[b].add_(band)
                    band_energy_sums[b].add_(band.flatten(1).norm(p=2, dim=1))
                    if need_consensus:
                        unit = _l2_unit(band)
                        unit_sums[b].add_(unit)
                        unit_squared_norm_sums[b].add_(unit.flatten(1).square().sum(dim=1))
                        del unit
                    del band
                del spectrum

            pool_count += 1
            loss_sum += float(loss.detach().cpu())
            del grad_n, grad, loss, logits, x_in, x_req

    if pool_count < 1:
        raise RuntimeError("SV-FCA source-view pool is empty")

    spatial_mean = spatial_sum.div(float(pool_count))
    result = {
        "spatial_mean": spatial_mean,
        "pool_count": pool_count,
        "source_names": source_names,
        "loss_mean": loss_sum / float(pool_count),
    }
    if need_frequency:
        result["band_means"] = [b.div(float(pool_count)) for b in band_sums]
        result["band_energy"] = torch.stack([e.div(float(pool_count)) for e in band_energy_sums], dim=1)
        if need_consensus:
            result["consensus"] = torch.stack(
                [
                    _pairwise_consensus_from_unit_sum(u, pool_count, diagonal)
                    for u, diagonal in zip(unit_sums, unit_squared_norm_sums)
                ], dim=1
            )
        else:
            result["consensus"] = torch.ones(
                x.shape[0], spectral_bands, device=x.device, dtype=torch.float32
            )
    return result


class SVFCAAttack(Attack):
    name = "sv_fca"

    def __call__(self, models, x, y):
        flags = _variant_flags(self.kwargs.get("variant", "full_model"), self.kwargs.get("num_views", 4))
        spectral_bands = int(self.kwargs.get("spectral_bands", self.kwargs.get("num_bands", 6)))
        diversity_prob = float(self.kwargs.get("diversity_prob", 1.0))
        image_size = int(self.kwargs.get("image_size", 224))
        resize_size = int(self.kwargs.get("resize_size", 256))
        decay = float(self.kwargs.get("decay", 1.0))
        band_temperature = float(self.kwargs.get("band_temperature", 0.20))
        low_mid_strength = float(self.kwargs.get("low_mid_strength", 1.0))
        spectral_decay = float(self.kwargs.get("spectral_decay", 0.65))
        consensus_gain = float(self.kwargs.get("consensus_gain", 2.0))
        energy_strength = float(self.kwargs.get("energy_strength", 0.75))
        weight_floor = float(self.kwargs.get("weight_floor", 0.02))
        amp = bool(self.kwargs.get("amp", False))
        amp_dtype = str(self.kwargs.get("amp_dtype", "fp16"))
        if not math.isfinite(band_temperature) or band_temperature <= 0:
            raise ValueError("band_temperature must be finite and positive")
        if not 0.0 <= spectral_decay < 1.0:
            raise ValueError("spectral_decay must be in [0, 1)")
        if not math.isfinite(low_mid_strength) or low_mid_strength < 0:
            raise ValueError("low_mid_strength must be finite and nonnegative")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and nonnegative")
        if not math.isfinite(consensus_gain) or consensus_gain < 0:
            raise ValueError("consensus_gain must be finite and nonnegative")
        if not math.isfinite(energy_strength) or energy_strength < 0:
            raise ValueError("energy_strength must be finite and nonnegative")
        if not math.isfinite(weight_floor) or not 0 <= weight_floor <= 1.0 / spectral_bands:
            raise ValueError("weight_floor must be in [0, 1 / spectral_bands]")
        if int(self.kwargs.get("num_views", 4)) < 1:
            raise ValueError("num_views must be positive")
        if amp_dtype not in ("fp16", "bf16"):
            raise ValueError("amp_dtype must be fp16 or bf16")

        x_clean = x.detach()
        x_adv = x_clean.clone()
        momentum = torch.zeros_like(x_adv)
        spectral_memory = None
        logs = []

        for step in range(self.steps):
            stats = _stream_source_view_frequency_stats(
                models=models,
                x=x_adv,
                y=y,
                num_views=flags["num_views"],
                spectral_bands=spectral_bands,
                diversity_prob=diversity_prob,
                image_size=image_size,
                resize_size=resize_size,
                need_frequency=flags["frequency_coordination"],
                need_consensus=flags["band_consensus"],
                amp=amp,
                amp_dtype=amp_dtype,
            )

            if not flags["frequency_coordination"]:
                direction = stats["spatial_mean"]
                instant_weights = None
                used_weights = None
                consensus = None
            else:
                consensus = stats["consensus"]
                instant_weights = _compute_band_weights(
                    consensus=consensus,
                    band_energy=stats["band_energy"],
                    band_temperature=band_temperature,
                    low_mid_strength=low_mid_strength,
                    consensus_gain=consensus_gain,
                    energy_strength=energy_strength,
                    weight_floor=weight_floor,
                    use_consensus=flags["band_consensus"],
                    use_prior=flags["low_mid_prior"],
                )

                if flags["spectral_memory"]:
                    if spectral_memory is None:
                        spectral_memory = instant_weights.detach()
                    else:
                        spectral_memory = (
                            spectral_decay * spectral_memory
                            + (1.0 - spectral_decay) * instant_weights.detach()
                        )
                        spectral_memory = spectral_memory / spectral_memory.sum(dim=1, keepdim=True).clamp_min(_EPS)
                    used_weights = spectral_memory
                else:
                    used_weights = instant_weights

                direction = torch.zeros_like(stats["spatial_mean"])
                for b, band_mean in enumerate(stats["band_means"]):
                    direction.add_(band_mean * used_weights[:, b].view(-1, 1, 1, 1))

            direction = normalize_grad_l1(direction)
            momentum.mul_(decay).add_(direction)
            x_adv = pgd_update(x_adv, x_clean, momentum, self.alpha, self.eps).detach()

            log = {
                "step": step,
                "algorithm": "SV-FCA",
                "variant": flags["variant"],
                "source_models": ",".join(stats["source_names"]),
                "num_sources": len(stats["source_names"]),
                "num_views": flags["num_views"],
                "pool_size": stats["pool_count"],
                "spectral_bands": spectral_bands,
                "loss_mean": stats["loss_mean"],
                "sv_pool_enabled": flags["num_views"] > 1,
                "frequency_coordination_enabled": flags["frequency_coordination"],
                "band_consensus_enabled": flags["band_consensus"],
                "low_mid_prior_enabled": flags["low_mid_prior"],
                "spectral_memory_enabled": flags["spectral_memory"],
                "band_temperature": band_temperature,
                "low_mid_strength": low_mid_strength,
                "spectral_decay": spectral_decay,
                "consensus_gain": consensus_gain,
                "energy_strength": energy_strength,
                "weight_floor": weight_floor,
                "token_branch_enabled": False,
                "target_access_during_attack": "none",
                "amp_enabled": amp,
            }
            if consensus is not None:
                c = consensus.detach().mean(dim=0).cpu().tolist()
                w = used_weights.detach().mean(dim=0).cpu().tolist()
                e = stats["band_energy"].detach().mean(dim=0).cpu().tolist()
                log["band_consensus"] = ";".join("{:.6f}".format(v) for v in c)
                log["band_energy"] = ";".join("{:.6f}".format(v) for v in e)
                log["band_weights"] = ";".join("{:.6f}".format(v) for v in w)
            logs.append(log)
            del stats, direction

        return AttackResult(x_adv.detach(), logs)


# Compatibility aliases for older project names.
SVFCA = SVFCAAttack
