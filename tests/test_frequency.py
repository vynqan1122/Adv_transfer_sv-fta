import math
import unittest

import torch

from attacks.sv_fca import _radial_rfft_masks
from src.frequency import _radial_mask, frequency_filter_fft


class FrequencyTests(unittest.TestCase):
    def test_fft_all_is_identity_for_even_and_odd_rectangles(self):
        for shape in ((2, 3, 8, 10), (2, 3, 7, 9)):
            x = torch.randn(shape, dtype=torch.float64)
            torch.testing.assert_close(frequency_filter_fft(x, mode="all"), x)

    def test_low_filter_preserves_low_sinusoid_on_even_grid(self):
        wave = torch.cos(2.0 * math.pi * torch.arange(8, dtype=torch.float64) / 8)
        x = wave.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        torch.testing.assert_close(frequency_filter_fft(x, mode="low"), x)

    def test_radial_mask_preserves_conjugate_symmetry(self):
        for h, w in ((8, 10), (7, 9)):
            mask = torch.fft.ifftshift(_radial_mask(h, w, "cpu"), dim=(-2, -1))
            reflected = mask.index_select(-2, (-torch.arange(h)) % h).index_select(-1, (-torch.arange(w)) % w)
            torch.testing.assert_close(mask, reflected)

    def test_radial_bands_partition_and_reconstruct(self):
        for h, w in ((8, 10), (7, 9)):
            masks = _radial_rfft_masks(h, w, 6, "cpu")
            torch.testing.assert_close(sum(masks), torch.ones(1, 1, h, w // 2 + 1))
            x = torch.randn(2, 3, h, w)
            spectrum = torch.fft.rfft2(x, norm="ortho")
            restored = sum(torch.fft.irfft2(spectrum * mask, s=(h, w), norm="ortho") for mask in masks)
            torch.testing.assert_close(restored, x)

    def test_high_filter_removes_dc(self):
        x = torch.ones(1, 3, 8, 8)
        torch.testing.assert_close(frequency_filter_fft(x, mode="high"), torch.zeros_like(x))

    def test_unknown_mode_is_not_silently_low_mid(self):
        with self.assertRaises(ValueError):
            frequency_filter_fft(torch.ones(1, 3, 8, 8), mode="typo")


if __name__ == "__main__":
    unittest.main()
