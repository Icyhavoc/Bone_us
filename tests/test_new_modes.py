"""Focused checks for channel 4 selection and probability voting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emd_pipeline import classification_metrics, prepare_frame_input, process_frames
from experiment_setup import CROSS_VOTE_REGION_NAME, region_experiments
from vit_model import RegionChannelCrossVoteViT, ViTConfig


def _sample(beam2_top: list[int]) -> np.ndarray:
    sample = np.zeros((50, 2, 20), dtype=np.float32)
    for rank, frame in enumerate(range(10)):
        sample[frame, 0, 0] = 20 - rank
    for rank, frame in enumerate(beam2_top):
        sample[frame, 1, 10] = 20 - rank
    return sample


def _span() -> dict[str, np.ndarray]:
    return {
        "starts": np.array([0, 10]),
        "ends": np.array([10, 20]),
    }


class Channel4Tests(unittest.TestCase):
    def _prepare(self, sample: np.ndarray, form: str, mode: int = 4):
        with patch("emd_pipeline.detect_dynamic_span", return_value=_span()):
            return prepare_frame_input(sample, form, mode)

    def test_max1_picks_best_beam1_frame_in_overlap(self) -> None:
        sample = _sample(list(range(5, 15)))
        processed, chosen, span = self._prepare(sample, "max1")
        self.assertEqual(chosen.tolist(), [5])
        np.testing.assert_array_equal(processed[0], sample[5])
        self.assertEqual(span["channel4_selection"]["overlap_frames_beam1_order"], [5, 6, 7, 8, 9])
        self.assertTrue(span["channel4_selection"]["used_overlap"])

    def test_top3_requires_three_common_frames(self) -> None:
        enough = _sample(list(range(5, 15)))
        processed, chosen, _ = self._prepare(enough, "top3_mean")
        self.assertEqual(chosen.tolist(), [5, 6, 7])
        np.testing.assert_allclose(processed[0], enough[[5, 6, 7]].mean(axis=0))

        only_two = _sample(list(range(8, 18)))
        _, fallback, span = self._prepare(only_two, "top3_mean")
        _, legacy = process_frames(only_two, "top3_mean", _span()["starts"], _span()["ends"])
        np.testing.assert_array_equal(fallback, legacy)
        self.assertFalse(span["channel4_selection"]["used_overlap"])

    def test_max1_no_overlap_keeps_legacy_similarity_fallback(self) -> None:
        sample = _sample(list(range(10, 20)))
        _, fallback, span = self._prepare(sample, "max1")
        _, legacy = process_frames(sample, "max1", _span()["starts"], _span()["ends"])
        np.testing.assert_array_equal(fallback, legacy)
        self.assertEqual(span["channel4_selection"]["fallback"], "channel3_similarity_ranking")

    def test_unranked_forms_keep_both_physical_channels(self) -> None:
        sample = _sample(list(range(10, 20)))
        for form in ("raw50", "mean_std"):
            channel4, chosen, _ = prepare_frame_input(sample, form, 4)
            channel3, _, _ = prepare_frame_input(sample, form, 3)
            self.assertIsNone(chosen)
            np.testing.assert_array_equal(channel4, channel3)


class Channel3Tests(unittest.TestCase):
    def _prepare(self, sample: np.ndarray, form: str):
        with patch("emd_pipeline.detect_dynamic_span", return_value=_span()):
            return prepare_frame_input(sample, form, 3)

    def test_independent_max1_with_different_frame_ids(self) -> None:
        sample = _sample(list(range(5, 15)))
        # A stronger out-of-window peak must not affect Beam2's main-window rank.
        sample[40, 1, 0] = 1000
        processed, chosen, span = self._prepare(sample, "max1")
        self.assertEqual(chosen.tolist(), [[0], [5]])
        self.assertEqual(processed.shape, (1, 2, 20))
        np.testing.assert_array_equal(processed[0, 0], sample[0, 0])
        np.testing.assert_array_equal(processed[0, 1], sample[5, 1])
        self.assertEqual(span["channel3_selection"]["selected_frames_by_channel"], [[0], [5]])

    def test_no_overlap_still_selects_each_beams_max1(self) -> None:
        processed, chosen, _ = self._prepare(_sample(list(range(10, 20))), "max1")
        self.assertEqual(chosen.tolist(), [[0], [10]])
        self.assertEqual(processed[0, 0, 0], 20)
        self.assertEqual(processed[0, 1, 10], 20)

    def test_top3_means_each_channel_independently(self) -> None:
        sample = _sample(list(range(5, 15)))
        processed, chosen, _ = self._prepare(sample, "top3_mean")
        self.assertEqual(chosen.tolist(), [[0, 1, 2], [5, 6, 7]])
        np.testing.assert_array_equal(processed[0, 0], sample[[0, 1, 2], 0].mean(0))
        np.testing.assert_array_equal(processed[0, 1], sample[[5, 6, 7], 1].mean(0))

    def test_ties_use_original_frame_order(self) -> None:
        sample = np.zeros((50, 2, 20), dtype=np.float32)
        sample[[3, 7], 0, 0] = 1
        sample[[4, 8], 1, 10] = 1
        _, chosen, _ = self._prepare(sample, "max1")
        self.assertEqual(chosen.tolist(), [[3], [4]])


class CrossVoteTests(unittest.TestCase):
    def test_three_regions_per_physical_channel(self) -> None:
        name, regions = region_experiments(CROSS_VOTE_REGION_NAME)[0]
        self.assertEqual(name, CROSS_VOTE_REGION_NAME)
        self.assertEqual([(region.start_mm, region.end_mm) for region in regions], [(0.0, 5.0)])

    def test_log_probability_matches_branch_vote(self) -> None:
        torch.manual_seed(7)
        config = ViTConfig(
            branches=6, streams=1, length=16, num_classes=2,
            patch_size=4, dim=8, depth=1, heads=2, mlp_dim=16, dropout=0.0,
        )
        model = RegionChannelCrossVoteViT(config).eval()
        signals = torch.randn(2, 6, 1, 16)
        with torch.no_grad():
            per_branch = model.branch_probabilities(signals)
            combined = model(signals).exp()
        torch.testing.assert_close(combined, per_branch.mean(dim=1))
        torch.testing.assert_close(per_branch.sum(dim=1), combined * 6)
        self.assertEqual(tuple(per_branch.shape), (2, 6, 2))

    def test_binary_vote_tie_uses_argmax(self) -> None:
        metrics = classification_metrics(
            np.array([0]), np.array([[0.5, 0.5]]), 2,
            binary_decision="argmax",
        )
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
