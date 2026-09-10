"""CPU regression checks; no pretrained weights or dataset downloads."""
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from attacks import build_attack
from attacks.base import normalize_grad_l1, smooth_grad_ti
from attacks.sinifgsm import SINIFGSM
from attacks.sv_fca import (
    _band_energy_features,
    _compute_band_weights,
    _l2_unit,
    _pairwise_consensus_from_unit_sum,
    _stream_source_view_frequency_stats,
)
from attacks.vit_aware import patch_saliency_from_grads
from src.models import NormalizedModel, create_model
from src.utils import read_csv, write_csv


class MeanClassifier(torch.nn.Module):
    def forward(self, x):
        s = x.mean(dim=(1, 2, 3))
        return torch.stack((s, -s), dim=1)


class AttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.rand(2, 3, 8, 8)
        self.y = torch.zeros(2, dtype=torch.long)
        self.models = {"toy": MeanClassifier()}

    def test_every_attack_respects_budget_and_increases_loss(self):
        for name in ("ifgsm", "mifgsm", "difgsm", "tifgsm", "si_ni_fgsm", "freq_only", "vit_aware", "sv_fca"):
            with self.subTest(attack=name):
                result = build_attack(
                    name, eps=0.1, alpha=0.06, steps=3, image_size=8,
                    resize_size=10, num_views=2, spectral_bands=3, diversity_prob=0.0,
                )(self.models, self.x, self.y)
                self.assertEqual(result.adv.shape, self.x.shape)
                self.assertFalse(result.adv.requires_grad)
                self.assertTrue(torch.isfinite(result.adv).all())
                self.assertGreaterEqual(result.adv.min().item(), 0.0)
                self.assertLessEqual(result.adv.max().item(), 1.0)
                self.assertLessEqual((result.adv - self.x).abs().max().item(), 0.100001)
                self.assertGreater(
                    F.cross_entropy(self.models["toy"](result.adv), self.y).item(),
                    F.cross_entropy(self.models["toy"](self.x), self.y).item(),
                )
                self.assertEqual(len(result.logs), 3)

    def test_zero_half_gradient_normalizes_without_nan(self):
        actual = normalize_grad_l1(torch.zeros_like(self.x, dtype=torch.float16))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.count_nonzero().item(), 0)

    def test_consensus_matches_explicit_pairs_including_zero_vectors(self):
        vectors = [torch.randn_like(self.x), torch.zeros_like(self.x), torch.randn_like(self.x)]
        units = [_l2_unit(g) for g in vectors]
        expected = sum(
            (units[i] * units[j]).flatten(1).sum(1)
            for i in range(len(units)) for j in range(i + 1, len(units))
        ) / 3.0
        diagonal = sum(u.flatten(1).square().sum(1) for u in units)
        actual = _pairwise_consensus_from_unit_sum(sum(units), len(units), diagonal)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_empty_frequency_bands_have_zero_pairwise_consensus(self):
        stats = _stream_source_view_frequency_stats(
            self.models, self.x, self.y, num_views=2, spectral_bands=3,
            diversity_prob=0.0, image_size=8, resize_size=10,
        )
        torch.testing.assert_close(stats["consensus"][:, 0], torch.ones(2))
        torch.testing.assert_close(stats["consensus"][:, 1:], torch.zeros(2, 2))
        torch.testing.assert_close(sum(stats["band_means"]), stats["spatial_mean"])

    def test_band_weighting_responds_to_consensus_and_energy(self):
        consensus = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        energy = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        uniform = _compute_band_weights(consensus, energy, 0.2, 0.0, energy_strength=0.0)
        selected = _compute_band_weights(consensus, energy, 0.2, 0.0, energy_strength=0.0)
        torch.testing.assert_close(uniform[0], torch.full((3,), 1 / 3), atol=1e-6, rtol=1e-6)
        self.assertGreater(selected[1, 0].item(), selected[1, 1].item())

        energy_skew = torch.tensor([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0]])
        energy_weights = _compute_band_weights(
            torch.zeros_like(consensus), energy_skew, 0.2, 0.0,
            energy_strength=1.0, use_prior=False,
        )
        self.assertGreater(energy_weights[0, 0].item(), energy_weights[0, 1].item())
        self.assertGreater(energy_weights[1, 1].item(), energy_weights[1, 0].item())
        self.assertTrue(torch.isfinite(_band_energy_features(energy_skew.unsqueeze(-1))).all())

    def test_si_scale_gradient_matches_differentiated_mean_loss(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(192, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
        scales = 3
        x_req = self.x.clone().requires_grad_(True)
        expected_loss = sum(F.cross_entropy(model(x_req / 2 ** i), self.y) for i in range(scales)) / scales
        expected = torch.autograd.grad(expected_loss, x_req)[0]
        grads, losses = SINIFGSM()._scaled_grad_per_model({"nonlinear": model}, self.x, self.y, scales)
        torch.testing.assert_close(grads[0], expected)
        self.assertAlmostEqual(losses[0], expected_loss.item(), places=6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_topk_saliency_selects_exact_count_on_ties(self):
        mask = patch_saliency_from_grads(["toy"], [torch.ones_like(self.x)], patch_size=2, hard_mask=True, topk_ratio=0.25)
        torch.testing.assert_close(mask.sum(dim=(1, 2, 3)), torch.full((2,), 16.0))

    def test_ti_preserves_shape_and_dtype(self):
        grad = self.x.double()
        out = smooth_grad_ti(grad, kernel_size=3, sigma=1.0)
        self.assertEqual(out.shape, grad.shape)
        self.assertEqual(out.dtype, grad.dtype)
        with self.assertRaises(ValueError):
            smooth_grad_ti(grad, kernel_size=4)

    def test_invalid_attack_parameters_fail_before_running(self):
        for kwargs in ({"eps": -1}, {"alpha": math.nan}, {"steps": 0}, {"steps": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_attack("ifgsm", **kwargs)


class ModelAndOutputTests(unittest.TestCase):
    def test_model_specific_preprocessing_is_differentiable(self):
        model = NormalizedModel(torch.nn.Identity(), mean=(0.5,) * 3, std=(0.5,) * 3, input_size=(3, 12, 12))
        x = torch.full((1, 3, 8, 8), 0.75, requires_grad=True)
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 12, 12))
        torch.testing.assert_close(out, torch.full_like(out, 0.5))
        grad = torch.autograd.grad(out.sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.min().item(), 0.0)

    def test_timm_checkpoint_metadata_drives_wrapper(self):
        inner = torch.nn.Identity()
        inner.pretrained_cfg = {"mean": (0.5,) * 3, "std": (0.5,) * 3, "input_size": (3, 12, 12)}
        fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: inner)
        with patch.dict("sys.modules", {"timm": fake_timm}):
            model = create_model("fake", torch.device("cpu"), pretrained=False)
        self.assertFalse(model.training)
        out = model(torch.ones(1, 3, 8, 8))
        torch.testing.assert_close(out, torch.ones(1, 3, 12, 12))

    def test_csv_keeps_columns_introduced_after_first_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv([{"model": "a"}, {"model": "b", "asr": 0.5}], path)
            rows = read_csv(path)
            self.assertEqual(rows[1]["asr"], "0.5")


if __name__ == "__main__":
    unittest.main()
