"""Train the waveform ViT using the project's existing split and preprocessing.

Example::

    python run_vit_experiments.py --data-dir raw_data_relabeled/cls2_thr1.3 \
        --forms max1 --region-set dyn_envelope --channel-mode 3
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dyn_cli import add_dyn_arguments, dyn_config_from_args
from emd_pipeline import (
    ADC_DC_MAGNITUDE,
    PREPROCESSING_TAG,
    PREPROCESSING_VERSION,
    DepthMapper,
    DynamicEnvelopeConfig,
    RegionSpec,
    classification_metrics,
    load_split,
    read_label_scheme,
    label_scheme_tag,
    resample_signals,
    write_json,
)
from experiment_setup import (
    DYNAMIC_REGION_NAME,
    REGION_CHOICES,
    guard_existing_experiment,
    infer_num_classes,
    parse_forms,
    region_experiments,
)
from vit_pipeline import build_vit_inputs

# This Conda workspace ships two OpenMP runtimes.  Initialize SciPy's batched
# spline solver before PyTorch so mean/std and raw50 can coexist with torch.
# This is the same cubic-spline path used for real inputs; it changes no data.
resample_signals(np.ones((4, 896), dtype=np.float32), 512)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vit_model import RegionChannelViT, ViTConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT classification of preprocessed BMU waveforms")
    parser.add_argument("--data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--forms", default="max1", help="all or comma-separated frame forms")
    parser.add_argument("--region-set", choices=REGION_CHOICES, default=DYNAMIC_REGION_NAME)
    parser.add_argument("--regions-json", default=None)
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--tukey-alpha", type=float, default=0.3)
    parser.add_argument("--max-depth-mm", type=float, default=5.0)
    parser.add_argument("--signal-length", type=int, default=896)
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    add_dyn_arguments(parser)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--positive-reward", type=float, default=2.0,
                        help="binary only: weight for label=1 recall in validation threshold selection")
    parser.add_argument("--no-reward", action="store_true",
                        help="binary only: use the original fixed 0.5 decision threshold")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="limit each split for a short smoke run")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.target_length) < 1:
        parser.error("epochs, patience, batch size and target length must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    if not math.isfinite(args.positive_reward) or args.positive_reward < 1.0:
        parser.error("--positive-reward must be finite and at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    return args


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else
                        "cpu" if value == "auto" else value)


def _load_prepared_split(
    data_dir: Path,
    split: str,
    *,
    form: str,
    regions: Sequence[RegionSpec],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    tukey_alpha: float,
    dynamic_config: DynamicEnvelopeConfig | None,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    raw, labels, records = load_split(data_dir, split)
    if max_samples is not None:
        raw, labels, records = raw[:max_samples], labels[:max_samples], records[:max_samples]
    inputs, audit = build_vit_inputs(
        raw, sample_ids=[str(record.get("sample_id", "")) for record in records],
        form=form, regions=regions, channel_mode=channel_mode, mapper=mapper,
        target_length=target_length, tukey_alpha=tukey_alpha,
        dynamic_envelope_config=dynamic_config,
    )
    return inputs, labels.astype(np.int64), records, audit


@torch.inference_mode()
def _evaluate(
    model: RegionChannelViT,
    inputs: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels)),
        batch_size=batch_size, shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    for waveforms, targets in loader:
        logits = model(waveforms.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("ViT produced non-finite logits")
        loss_sum += float(criterion(logits, targets.to(device))) * len(targets)
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities), loss_sum / len(labels)


def _fit(
    model: RegionChannelViT,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for waveforms, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms.to(device))
            loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("ViT training loss is non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        train_loss = loss_sum / len(train_y)
        val_probabilities, val_loss = _evaluate(
            model, val_x, val_y, criterion, args.batch_size, device,
        )
        val_metrics = classification_metrics(val_y, val_probabilities, model.config.num_classes)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_metrics["auc"], "val_accuracy": val_metrics["accuracy"],
        })
        print(f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no valid ViT checkpoint was produced")
    model.load_state_dict(best_state)
    return history, best_epoch, best_loss


def select_binary_threshold(
    labels: np.ndarray, probabilities: np.ndarray, positive_reward: float
) -> float:
    """Maximize specificity + reward * sensitivity on validation."""

    scores = np.asarray(probabilities[:, 1], dtype=np.float64)
    negatives = int(np.count_nonzero(labels == 0))
    positives = int(np.count_nonzero(labels == 1))
    if negatives == 0 or positives == 0:
        raise ValueError("validation must contain both labels to select a normalized threshold")
    candidates = np.r_[0.0, np.unique(scores), np.nextafter(scores.max(), np.inf)]
    best_score = -math.inf
    best_threshold = 0.0
    for threshold in candidates:
        predicted_positive = scores >= threshold
        tn = int(np.count_nonzero((labels == 0) & ~predicted_positive))
        tp = int(np.count_nonzero((labels == 1) & predicted_positive))
        score = tn / negatives + positive_reward * tp / positives
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def _binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float,
    positive_reward: float | None,
) -> dict[str, Any]:
    result = classification_metrics(labels, probabilities, 2, threshold=threshold)
    weight = 1.0 if positive_reward is None else positive_reward
    negatives = int(result["tn"]) + int(result["fp"])
    positives = int(result["fn"]) + int(result["tp"])
    result["weighted_balanced_accuracy"] = (
        (result["specificity"] + weight * result["sensitivity"]) / (1.0 + weight)
        if negatives and positives else None
    )
    result["decision_threshold"] = threshold
    result["positive_reward"] = positive_reward
    result["decision_mode"] = "fixed_0.5" if positive_reward is None else "normalized_reward"
    return result


def _experiment_name(
    form: str, region: str, channel: int, tag: str | None, decision_tag: str | None
) -> str:
    name = f"form_{form}__regions_{region}__channels_{channel}__vit__{PREPROCESSING_TAG}"
    if tag is not None:
        name = f"{name}__{tag}"
    return name if decision_tag is None else f"{name}__{decision_tag}"


def run_one(
    data_dir: Path,
    output_dir: Path,
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    args: argparse.Namespace,
    num_classes: int,
    label_scheme: dict[str, Any] | None,
    decision_tag: str | None,
) -> dict[str, Any]:
    mapper = DepthMapper(args.max_depth_mm, args.signal_length, args.rounding)
    dynamic_config = dyn_config_from_args(args) if region_name == DYNAMIC_REGION_NAME else None
    streams = 2 if form == "mean_std" else 50 if form == "raw50" else 1
    branches = (2 if dynamic_config is not None else len(regions)) * (2 if args.channel_mode == 3 else 1)
    vit_config = ViTConfig(
        branches=branches, streams=streams, length=args.target_length,
        num_classes=num_classes, patch_size=args.patch_size, dim=args.dim,
        depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    vit_config.validate()
    config: dict[str, Any] = {
        "model_family": "vit", "feature_layout": PREPROCESSING_TAG,
        "form": form, "region_name": region_name, "regions": regions,
        "channel_mode": args.channel_mode, "target_length": args.target_length,
        "tukey_alpha": args.tukey_alpha, "dynamic_envelope": dynamic_config,
        "depth_mapper": mapper, "preprocessing_version": PREPROCESSING_VERSION,
        "vit": vit_config, "data_dir": data_dir, "num_classes": num_classes,
        "label_scheme": label_scheme, "adc_dc_replacement": ADC_DC_MAGNITUDE,
        "max_samples": args.max_samples,
        "training": {"epochs": args.epochs, "patience": args.patience,
                     "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "weight_decay": args.weight_decay, "seed": args.seed},
    }
    if num_classes == 2:
        config["decision"] = {
            "method": "fixed_0.5" if args.no_reward else "normalized_reward",
            "positive_reward": None if args.no_reward else args.positive_reward,
            "selection_split": None if args.no_reward else "val",
        }
        config["decision_tag"] = decision_tag
    guard_existing_experiment(output_dir, config)
    device = _device(args.device)
    _set_seed(args.seed)
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first_audit: list[dict[str, Any]] | None = None
    for split in ("train", "val"):
        inputs, labels, records, audit = _load_prepared_split(
            data_dir, split, form=form, regions=regions,
            channel_mode=args.channel_mode, mapper=mapper,
            target_length=args.target_length, tukey_alpha=args.tukey_alpha,
            dynamic_config=dynamic_config, max_samples=args.max_samples,
        )
        if inputs.shape[1:] != (branches, streams, args.target_length):
            raise ValueError(f"unexpected {split} ViT input shape: {inputs.shape}")
        data[split] = inputs, labels
        np.save(output_dir / f"labels_{split}.npy", labels)
        write_json(output_dir / f"samples_{split}.json", records)
        write_json(output_dir / f"input_audit_{split}.json", audit)
        if split == "train":
            first_audit = audit
        print(f"[{form}/{region_name}] prepared {split}: {inputs.shape}", flush=True)
    assert first_audit is not None
    write_json(output_dir / "branch_info.json", {
        "branch_names": first_audit[0]["branch_names"],
        "shape_without_batch": list(data["train"][0].shape[1:]),
        "stream_meaning": "mean,std" if form == "mean_std" else
                          "original 50 frames" if form == "raw50" else "one selected waveform",
    })
    model = RegionChannelViT(vit_config).to(device)
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    history, best_epoch, best_loss = _fit(model, train_x, train_y, val_x, val_y, args, device)
    criterion = nn.CrossEntropyLoss()
    val_probabilities: np.ndarray | None = None
    decision_threshold: float | None = None
    if num_classes == 2:
        decision_threshold = 0.5
        if not args.no_reward:
            val_probabilities, _ = _evaluate(model, val_x, val_y, criterion, args.batch_size, device)
            decision_threshold = select_binary_threshold(
                val_y, val_probabilities, args.positive_reward
            )
            selected_val_metrics = _binary_metrics(
                val_y, val_probabilities, decision_threshold, args.positive_reward
            )
            print(
                f"[{form}/{region_name}] val-selected threshold={decision_threshold:.4f} "
                f"weighted_balanced_accuracy={selected_val_metrics['weighted_balanced_accuracy']:.4f} "
                f"label1_recall={selected_val_metrics['sensitivity']:.4f}",
                flush=True,
            )
    torch.save({"model": model.cpu().state_dict(), "config": vit_config.__dict__,
                "best_epoch": best_epoch, "decision_threshold": decision_threshold,
                "positive_reward": args.positive_reward if num_classes == 2 and not args.no_reward else None},
               output_dir / "vit_model.pt")
    model.to(device)
    write_json(output_dir / "history.json", history)
    # The holdout is read only after validation has frozen the model weights.
    test_x, test_y, test_records, test_audit = _load_prepared_split(
        data_dir, "test", form=form, regions=regions, channel_mode=args.channel_mode,
        mapper=mapper, target_length=args.target_length, tukey_alpha=args.tukey_alpha,
        dynamic_config=dynamic_config, max_samples=args.max_samples,
    )
    if test_x.shape[1:] != train_x.shape[1:]:
        raise ValueError(f"unexpected test ViT input shape: {test_x.shape}")
    data["test"] = test_x, test_y
    np.save(output_dir / "labels_test.npy", test_y)
    write_json(output_dir / "samples_test.json", test_records)
    write_json(output_dir / "input_audit_test.json", test_audit)
    split_metrics: dict[str, Any] = {}
    for split, (inputs, labels) in data.items():
        probabilities = val_probabilities if split == "val" and val_probabilities is not None else None
        if probabilities is None:
            probabilities, _ = _evaluate(model, inputs, labels, criterion, args.batch_size, device)
        np.save(output_dir / f"probabilities_{split}.npy", probabilities)
        split_metrics[split] = (
            _binary_metrics(
                labels, probabilities, decision_threshold,
                None if args.no_reward else args.positive_reward,
            )
            if decision_threshold is not None else
            classification_metrics(labels, probabilities, num_classes)
        )
    metrics: dict[str, Any] = {
        "feature_dim": branches * args.dim,
        "feature_group_dims": [args.dim] * branches,
        "best_epoch": best_epoch, "trained_epochs": len(history),
        "best_val_loss": best_loss,
        "split_counts": {split: len(labels) for split, (_, labels) in data.items()},
        "num_classes": num_classes, "label_scheme": label_scheme,
        "decision_threshold": decision_threshold,
        "decision_tag": decision_tag,
        "class_counts": {
            split: {str(label): int(np.count_nonzero(labels == label))
                    for label in range(num_classes)}
            for split, (_, labels) in data.items()
        },
        "metrics": split_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    test_metrics = split_metrics["test"]
    report = f"[{form}/{region_name}] ViT test_accuracy={test_metrics['accuracy']:.4f}"
    if num_classes == 2:
        balanced = test_metrics["weighted_balanced_accuracy"]
        balanced_text = "n/a" if balanced is None else f"{balanced:.4f}"
        report += (f" weighted_balanced_accuracy={balanced_text}"
                   f" label1_recall={test_metrics['sensitivity']:.4f}")
    print(report, flush=True)
    return metrics


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    label_scheme = read_label_scheme(data_dir)
    tag = label_scheme_tag(label_scheme)
    inferred = infer_num_classes(data_dir, args.max_samples)
    recorded = int(label_scheme["num_classes"]) if label_scheme else 2
    num_classes = max(inferred, recorded) if args.num_classes is None else args.num_classes
    if num_classes < max(inferred, recorded):
        raise ValueError("--num-classes is smaller than the dataset label scheme")
    decision_tag = (
        ("fixed0p5" if args.no_reward else
         "norm_w1_" + str(args.positive_reward).replace(".", "p").replace("+", ""))
        if num_classes == 2 else None
    )
    forms = parse_forms(args.forms)
    regions_to_run = region_experiments(args.region_set, args.regions_json)
    started = time.time()
    results: dict[str, Any] = {}
    for form in forms:
        for region_name, regions in regions_to_run:
            name = _experiment_name(form, region_name, args.channel_mode, tag, decision_tag)
            results[name] = run_one(
                data_dir, output_dir / name, form, region_name, regions, args,
                num_classes, label_scheme, decision_tag,
            )
    write_json(output_dir / "summary_vit.json", {
        "data_dir": data_dir, "forms": forms,
        "regions": [name for name, _ in regions_to_run],
        "channel_mode": args.channel_mode, "decision_tag": decision_tag,
        "results": results,
        "elapsed_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
