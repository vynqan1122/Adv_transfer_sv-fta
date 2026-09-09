import math

import torch


def _radial_mask(height, width, device, mode="low_mid", agreement=0.0):
    # Use actual FFT bins. A linspace grid is offset on even-sized images and
    # breaks conjugate symmetry, so dropping the inverse FFT's imaginary part
    # would silently change the intended filter.
    yy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device)).view(height, 1)
    xx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device)).view(1, width)
    radius = torch.sqrt(xx * xx + yy * yy)
    radius = radius / math.sqrt(0.5 ** 2 + 0.5 ** 2)

    if mode == "low":
        mask = (radius <= 0.25).float()
    elif mode == "mid":
        mask = ((radius > 0.20) & (radius <= 0.55)).float()
    elif mode == "high":
        mask = (radius > 0.55).float()
    elif mode == "all":
        mask = torch.ones_like(radius)
    elif mode == "low_mid":
        # low_mid: favor low + mid. If agreement is low, suppress high freq more.
        agreement_01 = float(max(0.0, min(1.0, (agreement + 1.0) / 2.0)))
        low = (radius <= 0.25).float() * 1.00
        mid = ((radius > 0.25) & (radius <= 0.55)).float() * 0.75
        high_weight = 0.10 + 0.25 * agreement_01
        high = (radius > 0.55).float() * high_weight
        mask = low + mid + high
    else:
        raise ValueError("Unknown frequency mode {!r}; choose low, mid, high, all, or low_mid".format(mode))
    return mask.view(1, 1, height, width)


def frequency_filter_fft(grad, mode="low_mid", agreement=0.0):
    """Filter real image gradients using a radial, conjugate-symmetric FFT mask."""
    g = grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
    h, w = g.shape[-2], g.shape[-1]
    spectrum = torch.fft.fft2(g, dim=(-2, -1))
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    mask = _radial_mask(h, w, g.device, mode=mode, agreement=agreement)
    filtered = spectrum * mask
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    out = torch.fft.ifft2(filtered, dim=(-2, -1)).real
    return out.to(dtype=grad.dtype)
