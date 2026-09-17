"""Shared command-line plumbing for the ``dyn_envelope`` region.

The detector constants live in :class:`emd_pipeline.DynamicEnvelopeConfig`, so
this module only mirrors that dataclass onto ``argparse``.  Every entry point
(training and both visualizers) uses these helpers, which keeps the three
scripts from drifting apart and keeps the dataclass the single source of
defaults.
"""

from __future__ import annotations

import argparse

from emd_pipeline import DynamicEnvelopeConfig


def add_dyn_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``--dyn-*`` options that configure the envelope split."""

    defaults = DynamicEnvelopeConfig()
    parser.add_argument(
        "--dyn-top-k",
        type=int,
        default=defaults.locator_top_k,
        help="frames averaged per channel to build the locator signal",
    )
    parser.add_argument(
        "--dyn-noise-start",
        type=int,
        default=defaults.noise_start,
        help="first sample considered by the crossing detector",
    )
    parser.add_argument(
        "--dyn-weak-threshold",
        type=float,
        default=defaults.weak_threshold,
        help="envelope level of the first reference crossing",
    )
    parser.add_argument(
        "--dyn-smooth-window",
        type=int,
        default=defaults.smooth_window,
        help="moving-average window for the envelope detector",
    )
    parser.add_argument(
        "--dyn-peak-window-back",
        type=int,
        default=defaults.peak_window_back,
        help="samples searched before the reference crossing for the local peak",
    )
    parser.add_argument(
        "--dyn-peak-window-forward",
        type=int,
        default=defaults.peak_window_forward,
        help="samples searched after the reference crossing for the local peak",
    )
    parser.add_argument(
        "--dyn-peak-ratio",
        type=float,
        default=defaults.peak_ratio,
        help="run threshold as a fraction of the local envelope peak",
    )
    parser.add_argument(
        "--dyn-gap-max",
        type=int,
        default=defaults.gap_max,
        help="largest gap closed while merging the leading packet",
    )
    parser.add_argument(
        "--dyn-min-run-width",
        type=int,
        default=defaults.min_run_width,
        help="minimum width of a mergeable run",
    )
    parser.add_argument(
        "--dyn-min-run-area-ratio",
        type=float,
        default=defaults.min_run_area_ratio,
        help="minimum run area as a fraction of the anchor run",
    )
    parser.add_argument(
        "--dyn-merge-max-lead",
        type=int,
        default=defaults.merge_max_lead,
        help="maximum total lead of the merged packet",
    )
    parser.add_argument(
        "--dyn-lead-back",
        type=int,
        default=defaults.lead_back,
        help="samples subtracted from the threshold index to start the main window",
    )
    parser.add_argument(
        "--dyn-main-start-min",
        type=int,
        default=defaults.main_start_min,
        help="lower clamp of the main window start",
    )
    parser.add_argument(
        "--dyn-main-length",
        type=int,
        default=defaults.main_length,
        help="fixed length of the main branch",
    )
    parser.add_argument(
        "--dyn-tail-start",
        type=int,
        default=defaults.tail_start,
        help="fixed start index of the tail branch",
    )
    parser.add_argument(
        "--dyn-no-manual-corrections",
        action="store_true",
        help="skip the hand-checked per-channel corrections ported from the reference",
    )


def dyn_config_from_args(args: argparse.Namespace) -> DynamicEnvelopeConfig:
    """Build the detector configuration from parsed ``--dyn-*`` options."""

    return DynamicEnvelopeConfig(
        noise_start=args.dyn_noise_start,
        weak_threshold=args.dyn_weak_threshold,
        smooth_window=args.dyn_smooth_window,
        peak_window_back=args.dyn_peak_window_back,
        peak_window_forward=args.dyn_peak_window_forward,
        peak_ratio=args.dyn_peak_ratio,
        gap_max=args.dyn_gap_max,
        min_run_width=args.dyn_min_run_width,
        min_run_area_ratio=args.dyn_min_run_area_ratio,
        merge_max_lead=args.dyn_merge_max_lead,
        lead_back=args.dyn_lead_back,
        main_start_min=args.dyn_main_start_min,
        main_length=args.dyn_main_length,
        tail_start=args.dyn_tail_start,
        locator_top_k=args.dyn_top_k,
        apply_manual_corrections=not args.dyn_no_manual_corrections,
    )
