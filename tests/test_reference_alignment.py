from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emd_pipeline import (
    REFERENCE_SHARED_FRAME_SELECTION,
    RESAMPLING_CUBIC,
    RESAMPLING_REFERENCE_LINEAR,
    DynamicEnvelopeConfig,
    prepare_frame_input,
    resample_signals,
)
from run_reference_aligned_cv import current_label_metrics, reference_lr
import torch
from vit_model import RegionChannelBeamVoteViT, ViTConfig


class SharedPeakSelectionTests(unittest.TestCase):
    def test_shared_peak_uses_full_two_beam_trace(self):
        sample = np.zeros((50, 2, 896), dtype=np.float32)
        sample[7, 0, 500] = 3.0
        sample[11, 1, 700] = -5.0
        processed, selected, span = prepare_frame_input(
            sample, "max1", 3, "synthetic", DynamicEnvelopeConfig(),
            REFERENCE_SHARED_FRAME_SELECTION,
        )
        self.assertEqual(selected.tolist(), [11])
        np.testing.assert_array_equal(processed[0], sample[11])
        self.assertEqual(
            span["shared_frame_selection"]["algorithm"],
            REFERENCE_SHARED_FRAME_SELECTION,
        )

    def test_reference_linear_resampling_matches_np_interp(self):
        signals = np.asarray([[0.0, 2.0, -1.0], [3.0, 1.0, 4.0]], dtype=np.float32)
        expected = np.stack([
            np.interp(np.linspace(0, 2, 7), np.arange(3), row)
            for row in signals
        ]).astype(np.float32)
        actual = resample_signals(signals, 7, RESAMPLING_REFERENCE_LINEAR)
        np.testing.assert_array_equal(actual, expected)

    def test_cubic_remains_the_default_resampling(self):
        signals = np.asarray([[0.0, 2.0, -1.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            resample_signals(signals, 9),
            resample_signals(signals, 9, RESAMPLING_CUBIC),
        )


class BeamVoteModelTests(unittest.TestCase):
    def test_each_head_reads_all_regions_of_one_beam(self):
        torch.manual_seed(3)
        model = RegionChannelBeamVoteViT(ViTConfig(
            branches=6, streams=1, length=16, num_classes=2,
            patch_size=4, dim=8, depth=1, heads=2, mlp_dim=16, dropout=0,
        )).eval()
        signal = torch.randn(2, 6, 1, 16)
        with torch.inference_mode():
            original = model.beam_logits(signal)
            changed = signal.clone()
            changed[:, 0] += 2.0  # main Beam1 only
            updated = model.beam_logits(changed)
        self.assertFalse(torch.equal(original[:, 0], updated[:, 0]))
        torch.testing.assert_close(original[:, 1], updated[:, 1], rtol=0, atol=0)

    def test_forward_is_log_of_two_beam_probability_mean(self):
        torch.manual_seed(4)
        model = RegionChannelBeamVoteViT(ViTConfig(
            branches=4, streams=1, length=16, num_classes=2,
            patch_size=4, dim=8, depth=1, heads=2, mlp_dim=16, dropout=0,
        )).eval()
        signal = torch.randn(3, 4, 1, 16)
        with torch.inference_mode():
            expected = model.beam_probabilities(signal).mean(1).log()
            actual = model(signal)
        torch.testing.assert_close(actual, expected)

    def test_reference_region_layout_matches_combined_layout(self):
        torch.manual_seed(5)
        model = RegionChannelBeamVoteViT(ViTConfig(
            branches=6, streams=1, length=16, num_classes=2,
            patch_size=4, dim=8, depth=1, heads=2, mlp_dim=16, dropout=0,
        )).eval()
        regions = tuple(torch.randn(3, 2, 16) for _ in range(3))
        combined = torch.stack([
            region[:, beam] for region in regions for beam in range(2)
        ], dim=1)[:, :, None, :]
        with torch.inference_mode():
            expected = model.beam_logits(combined)
            actual = model.beam_logits_from_regions(*regions)
        torch.testing.assert_close(actual, expected)


class ReferenceTrainingProtocolTests(unittest.TestCase):
    def test_reference_learning_rate_matches_warmup_and_zero_endpoint(self):
        self.assertAlmostEqual(reference_lr(1, peak=3e-4, warmup=5, epochs=120), 6e-5)
        self.assertAlmostEqual(reference_lr(5, peak=3e-4, warmup=5, epochs=120), 3e-4)
        self.assertAlmostEqual(reference_lr(120, peak=3e-4, warmup=5, epochs=120), 0.0)

    def test_reference_probabilities_are_mapped_back_to_project_labels(self):
        current_labels = np.asarray([0, 1])  # thin, thick
        reference_probabilities = np.asarray([[0.1, 0.9], [0.8, 0.2]])  # thick, thin
        metrics = current_label_metrics(current_labels, reference_probabilities)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
