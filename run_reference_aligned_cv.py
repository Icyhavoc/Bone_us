"""Reference-aligned six-branch Beam vote with selectable resampling.

The default retains the project SciPy CubicSpline preprocessing.  Explicit
``--resampling reference_linear`` uses the frozen reference ``np.interp``
implementation and requires the prepared tensors to match its caches exactly.
Both modes use the reference shared full-trace peak frame, five folds, two Beam
heads, mean per-Beam cross entropy, validation-AUC checkpointing, BF16, fused
AdamW, CUDA randperm, and five-fold probability ensembling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from emd_pipeline import (
    REFERENCE_SHARED_FRAME_SELECTION,
    RESAMPLING_CUBIC,
    RESAMPLING_REFERENCE_LINEAR,
    DepthMapper,
    DynamicEnvelopeConfig,
    RegionSpec,
    classification_metrics,
    load_split,
)
from vit_pipeline import build_vit_inputs

# Keep the shared SciPy initialization ahead of torch, matching the main entry.
from emd_pipeline import resample_signals
resample_signals(np.ones((4, 896), dtype=np.float32), 512)

import torch
from torch.nn import functional as F

from vit_model import RegionChannelBeamVoteViT, ViTConfig


APP_DIR = Path(__file__).resolve().parent
SOURCE_FILES = (
    "run_reference_aligned_cv.py", "emd_pipeline.py", "vit_pipeline.py", "vit_model.py",
)
REFERENCE_PARAMETERS = {
    "patch_size": 8, "dim": 256, "depth": 2, "heads": 4, "mlp_dim": 1024,
    "dropout": 0.4, "learning_rate": 3e-4, "weight_decay": 0.0,
    "batch_size": 4, "warmup_epochs": 5, "epochs": 120, "patience": 20,
    "seed": 2023,
}


def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # The frozen trainer resets these before enabling the automatic CUDA paths.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def reference_lr(epoch: int, *, peak: float, warmup: int, epochs: int) -> float:
    factor = (
        epoch / max(1, warmup)
        if epoch <= warmup else
        0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    )
    return peak * factor


def model_probabilities(
    model: RegionChannelBeamVoteViT,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int | None = 32,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ensemble, beams = [], []
    sample_count = len(inputs[0])
    batch_size = sample_count if batch_size is None else batch_size
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.beam_logits_from_regions(*[
                    region[start:start + batch_size] for region in inputs
                ]).float()
            probabilities = logits.softmax(dim=-1)
            beams.append(probabilities.cpu().numpy())
            ensemble.append(probabilities.mean(dim=1).cpu().numpy())
    return np.concatenate(ensemble), np.concatenate(beams)


def reference_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(labels, probabilities, 2, 0.5, binary_decision="threshold")
    metrics["correct"] = int(round(metrics["accuracy"] * len(labels)))
    metrics["n"] = int(len(labels))
    metrics["threshold"] = 0.5
    metrics["confusion_matrix"] = [
        [int(metrics["tn"]), int(metrics["fp"])],
        [int(metrics["fn"]), int(metrics["tp"])],
    ]
    return metrics


def current_label_metrics(current_labels: np.ndarray, model_probabilities: np.ndarray) -> dict[str, Any]:
    # Reference model: 0=thick, 1=thin. Project: 0=thin, 1=thick.
    current_probabilities = model_probabilities[:, [1, 0]]
    metrics = reference_metrics(current_labels, current_probabilities)
    metrics["label_semantics"] = {"0": "thin <1.3mm", "1": "thick >=1.3mm"}
    return metrics


def collect_dataset(data_dir: Path) -> dict[str, tuple[np.ndarray, int, dict[str, Any]]]:
    collected: dict[str, tuple[np.ndarray, int, dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        signals, labels, records = load_split(data_dir, split)
        for signal, label, record in zip(signals, labels, records):
            point_id = str(record["sample_id"])
            if point_id in collected:
                raise ValueError(f"duplicate point across current splits: {point_id}")
            collected[point_id] = signal, int(label), record
    return collected


def ordered_samples(
    collected: dict[str, tuple[np.ndarray, int, dict[str, Any]]], point_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    missing = [point_id for point_id in point_ids if point_id not in collected]
    if missing:
        raise ValueError(f"current unified dataset is missing point IDs: {missing[:5]}")
    return (
        np.stack([collected[point_id][0] for point_id in point_ids]),
        np.asarray([collected[point_id][1] for point_id in point_ids], dtype=np.int64),
        [collected[point_id][2] for point_id in point_ids],
    )


def prepare_inputs(options: argparse.Namespace, output: Path) -> dict[str, Any]:
    reference_data = options.reference_root / "top15_long_training_20261026/data"
    reference_input = reference_data / "methods/six_channel_beam_vote/input"
    reference_test = reference_data / "test_inputs/six_channel_beam_vote"
    development_meta = read_json(reference_input / "development.json")
    test_meta = read_json(reference_test / "test_data.json")
    development_ids = [str(value) for value in development_meta["point_ids"]]
    test_ids = [str(value) for value in test_meta["point_ids"]]
    if set(development_ids) & set(test_ids) or len(set(development_ids + test_ids)) != 268:
        raise ValueError("reference development/test point protocol is not a unique 268-point partition")

    collected = collect_dataset(options.data_dir)
    development_x, development_current_y, development_records = ordered_samples(collected, development_ids)
    test_x, test_current_y, test_records = ordered_samples(collected, test_ids)
    dynamic = DynamicEnvelopeConfig()
    regions = [RegionSpec(0.0, 5.0, "full")]
    development_inputs, development_audit = build_vit_inputs(
        development_x, sample_ids=development_ids, form="max1", regions=regions,
        channel_mode=3, mapper=DepthMapper(), target_length=512, tukey_alpha=0.3,
        dynamic_envelope_config=dynamic,
        frame_selection=REFERENCE_SHARED_FRAME_SELECTION,
        resampling_method=options.resampling,
    )
    test_inputs, test_audit = build_vit_inputs(
        test_x, sample_ids=test_ids, form="max1", regions=regions,
        channel_mode=3, mapper=DepthMapper(), target_length=512, tukey_alpha=0.3,
        dynamic_envelope_config=dynamic,
        frame_selection=REFERENCE_SHARED_FRAME_SELECTION,
        resampling_method=options.resampling,
    )
    if development_inputs.shape != (201, 6, 1, 512) or test_inputs.shape != (67, 6, 1, 512):
        raise ValueError(
            f"unexpected prepared shapes: {development_inputs.shape}, {test_inputs.shape}"
        )

    with np.load(reference_input / "development.npz", allow_pickle=False) as cached:
        selected = np.asarray([row["selected_frames"][0] for row in development_audit])
        if not np.array_equal(selected, cached["selected_frame_indices"][:, 0]):
            raise ValueError("development shared-peak frames differ from the frozen reference")
        model_y = 1 - development_current_y
        if not np.array_equal(model_y, cached["label"]):
            raise ValueError("development model labels differ from the frozen reference")
        if options.resampling == RESAMPLING_REFERENCE_LINEAR:
            cached_inputs = np.stack([
                cached["main"][:, 0], cached["main"][:, 1],
                cached["tail"][:, 0], cached["tail"][:, 1],
                cached["whole"][:, 0], cached["whole"][:, 1],
            ], axis=1)[:, :, None, :]
            np.testing.assert_array_equal(development_inputs, cached_inputs)
    with np.load(reference_test / "test_arrays.npz", allow_pickle=False) as cached:
        selected = np.asarray([row["selected_frames"][0] for row in test_audit])
        if not np.array_equal(selected, cached["selected_frame_indices"][:, 0]):
            raise ValueError("test shared-peak frames differ from the frozen reference")
        test_model_y = 1 - test_current_y
        if not np.array_equal(test_model_y, cached["label"]):
            raise ValueError("test model labels differ from the frozen reference")
        if options.resampling == RESAMPLING_REFERENCE_LINEAR:
            cached_inputs = np.stack([
                cached["main"][:, 0], cached["main"][:, 1],
                cached["tail"][:, 0], cached["tail"][:, 1],
                cached["whole"][:, 0], cached["whole"][:, 1],
            ], axis=1)[:, :, None, :]
            np.testing.assert_array_equal(test_inputs, cached_inputs)

    np.savez_compressed(
        output / f"prepared_{options.resampling}_inputs.npz",
        development=development_inputs, development_current_label=development_current_y,
        development_model_label=1 - development_current_y,
        test=test_inputs, test_current_label=test_current_y, test_model_label=1 - test_current_y,
    )
    atomic_json(output / "development_records.json", development_records)
    atomic_json(output / "test_records.json", test_records)
    atomic_json(output / "input_audit_development.json", development_audit)
    atomic_json(output / "input_audit_test.json", test_audit)
    return {
        "development": development_inputs,
        "development_current_y": development_current_y,
        "development_model_y": 1 - development_current_y,
        "development_ids": development_ids,
        "test": test_inputs,
        "test_current_y": test_current_y,
        "test_model_y": 1 - test_current_y,
        "test_ids": test_ids,
        "fold_path": reference_input / "fold_indices.npz",
    }


class AlignedRun:
    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.output = options.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output / "progress.json"
        reference_config = options.reference_root / "reference/configs/six_channel_beam_vote/random_0086.json"
        if not reference_config.is_file():
            raise FileNotFoundError(reference_config)
        frozen = read_json(reference_config)
        expected = {
            "patch_size": 8, "dim": 256, "depth": 2, "heads": 4,
            "mlp_dim": 1024, "epochs": 120, "patience": 20, "batch_size": 4,
            "warmup": 5, "dropout": 0.4, "lr": 0.0003, "weight_decay": 0.0,
        }
        if any(frozen[key] != value for key, value in expected.items()):
            raise ValueError("reference random_0086 parameters differ from the aligned protocol")
        manifest = {
            "created_for": f"full reference alignment with {options.resampling} resampling",
            "data_dir": str(options.data_dir.resolve()),
            "reference_root": str(options.reference_root.resolve()),
            "reference_config": str(reference_config.resolve()),
            "reference_config_sha256": sha256(reference_config),
            "parameters": {**REFERENCE_PARAMETERS, "epochs": options.epochs,
                           "patience": options.patience},
            "folds": options.folds,
            "preprocessing": {
                "frame_selection": REFERENCE_SHARED_FRAME_SELECTION,
                "regions": ["dynamic_main", "dynamic_tail", "full"],
                "resampling": options.resampling,
                "target_length": 512, "tukey_alpha": 0.3,
            },
            "model": "six_region_beam_vit_two_mlp_soft_vote",
            "loss": "mean_of_two_beam_cross_entropies",
            "checkpoint_selection": "fold_validation_auc",
            "ensemble": "equal_probability_mean_across_two_beams_and_five_folds",
            "model_labels": {"0": "thick >=1.3mm", "1": "thin <1.3mm"},
            "project_labels": {"0": "thin <1.3mm", "1": "thick >=1.3mm"},
            "source_sha256": {
                name: sha256(APP_DIR / name) for name in SOURCE_FILES
            },
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("output contains a different protocol; choose another output directory")
        atomic_json(manifest_path, manifest)
        source_dir = self.output / "sources"
        source_dir.mkdir(exist_ok=True)
        for name in SOURCE_FILES:
            shutil.copy2(APP_DIR / name, source_dir / name)
        self.progress = {
            "status": "preparing", "pid": os.getpid(), "started_at": stamp(),
            "fold": None, "epoch": 0, "epochs": options.epochs,
        }
        self.update()

    def update(self, **fields: Any) -> None:
        self.progress.update(fields, updated_at=stamp())
        atomic_json(self.progress_path, self.progress)

    def fit_fold(
        self, fold: int, train_rows: np.ndarray, val_rows: np.ndarray,
        development: np.ndarray, labels: np.ndarray,
    ) -> dict[str, Any]:
        directory = self.output / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        complete_path = directory / "complete.json"
        if complete_path.exists():
            complete = read_json(complete_path)
            if sha256(directory / "best.pt") != complete["checkpoint_sha256"]:
                raise ValueError(f"fold {fold} checkpoint hash mismatch")
            return complete

        config = ViTConfig(
            branches=6, streams=1, length=512, num_classes=2,
            patch_size=REFERENCE_PARAMETERS["patch_size"], dim=REFERENCE_PARAMETERS["dim"],
            depth=REFERENCE_PARAMETERS["depth"], heads=REFERENCE_PARAMETERS["heads"],
            mlp_dim=REFERENCE_PARAMETERS["mlp_dim"], dropout=REFERENCE_PARAMETERS["dropout"],
        )
        regions = tuple(
            np.ascontiguousarray(development[:, start:start + 2, 0, :])
            for start in (0, 2, 4)
        )
        train_x = tuple(torch.as_tensor(region[train_rows], device="cuda") for region in regions)
        val_x = tuple(torch.as_tensor(region[val_rows], device="cuda") for region in regions)
        train_y = torch.as_tensor(labels[train_rows], device="cuda")
        val_y = labels[val_rows]
        set_seed(REFERENCE_PARAMETERS["seed"])
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        model = RegionChannelBeamVoteViT(config).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=REFERENCE_PARAMETERS["learning_rate"],
            weight_decay=REFERENCE_PARAMETERS["weight_decay"], fused=True,
        )
        history: list[dict[str, Any]] = []
        best_auc = -1.0
        best_epoch = 0
        stale = 0
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        for epoch in range(1, self.options.epochs + 1):
            if stale >= self.options.patience:
                break
            learning_rate = reference_lr(
                epoch, peak=REFERENCE_PARAMETERS["learning_rate"],
                warmup=REFERENCE_PARAMETERS["warmup_epochs"], epochs=self.options.epochs,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            total_loss = 0.0
            order = torch.randperm(len(train_rows), device="cuda")
            for rows in order.split(REFERENCE_PARAMETERS["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.beam_logits_from_regions(*[
                        region[rows] for region in train_x
                    ])
                    loss = (
                        F.cross_entropy(logits[:, 0], train_y[rows])
                        + F.cross_entropy(logits[:, 1], train_y[rows])
                    ) / 2.0
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Beam vote loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.item()) * len(rows)
            val_probabilities, _ = model_probabilities(model, val_x, batch_size=None)
            metrics = reference_metrics(val_y, val_probabilities)
            auc = float(metrics["auc"])
            improved = auc > best_auc + 1e-8
            if improved:
                best_auc, best_epoch, stale = auc, epoch, 0
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                atomic_torch_save({
                    "model": state, "config": asdict(config), "fold": fold,
                    "epoch": epoch, "validation_auc": best_auc,
                    "model_labels": {"0": "thick", "1": "thin"},
                }, directory / "best.pt")
            else:
                stale += 1
            row = {
                "epoch": epoch, "train_loss": total_loss / len(train_rows),
                "validation_auc": auc, "validation_accuracy_05": metrics["accuracy"],
                "learning_rate": learning_rate, "best_epoch": best_epoch, "stale": stale,
            }
            history.append(row)
            atomic_json(directory / "history.json", history)
            self.update(status="training", fold=fold, epoch=epoch, best_epoch=best_epoch,
                        best_validation_auc=best_auc, stale=stale)
            print(
                f"fold={fold} epoch={epoch:03d} lr={learning_rate:.8g} "
                f"loss={row['train_loss']:.5f} val_auc={auc:.5f} "
                f"val_acc={metrics['accuracy']:.5f} best={best_epoch}", flush=True,
            )

        checkpoint = torch.load(directory / "best.pt", map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        val_probabilities, val_beams = model_probabilities(model, val_x, batch_size=None)
        np.save(directory / "validation_probabilities.npy", val_probabilities)
        np.save(directory / "validation_beam_probabilities.npy", val_beams)
        complete = {
            "fold": fold, "best_epoch": best_epoch, "epochs_run": len(history),
            "validation": reference_metrics(val_y, val_probabilities),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": sha256(directory / "best.pt"),
        }
        atomic_json(complete_path, complete)
        del model, optimizer, train_x, train_y, val_x
        torch.cuda.empty_cache()
        return complete

    def run(self) -> None:
        self.update(status="preprocessing", fold=None, epoch=0)
        data = prepare_inputs(self.options, self.output)
        development = data["development"]
        labels = data["development_model_y"]
        oof = np.full((len(labels), 2), np.nan, dtype=np.float32)
        fold_results = []
        with np.load(data["fold_path"], allow_pickle=False) as fold_file:
            for fold in self.options.folds:
                train_rows = fold_file[f"fold{fold}_train"]
                val_rows = fold_file[f"fold{fold}_validation"]
                result = self.fit_fold(fold, train_rows, val_rows, development, labels)
                fold_results.append(result)
                oof[val_rows] = np.load(self.output / f"fold_{fold}/validation_probabilities.npy")
        if self.options.folds == [1, 2, 3, 4, 5]:
            if not np.isfinite(oof).all():
                raise ValueError("five-fold OOF predictions are incomplete")
            np.save(self.output / "oof_probabilities.npy", oof)
            oof_metrics = reference_metrics(labels, oof)
        else:
            oof_metrics = None

        test_regions = tuple(
            np.ascontiguousarray(data["test"][:, start:start + 2, 0, :])
            for start in (0, 2, 4)
        )
        fold_probabilities, fold_beams = [], []
        for fold in self.options.folds:
            checkpoint = torch.load(
                self.output / f"fold_{fold}/best.pt", map_location="cuda", weights_only=True,
            )
            set_seed(REFERENCE_PARAMETERS["seed"])
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            model = RegionChannelBeamVoteViT(ViTConfig(**checkpoint["config"])).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=REFERENCE_PARAMETERS["learning_rate"],
                weight_decay=REFERENCE_PARAMETERS["weight_decay"], fused=True,
            )
            model.load_state_dict(checkpoint["model"])
            probabilities_parts, beam_parts = [], []
            for start in range(0, len(data["test"]), 32):
                batch = tuple(
                    torch.as_tensor(region[start:start + 32], device="cuda")
                    for region in test_regions
                )
                probabilities, beams = model_probabilities(model, batch, batch_size=None)
                probabilities_parts.append(probabilities)
                beam_parts.append(beams)
            probabilities = np.concatenate(probabilities_parts)
            beams = np.concatenate(beam_parts)
            fold_probabilities.append(probabilities)
            fold_beams.append(beams)
            del model, optimizer
            torch.cuda.empty_cache()
        by_fold = np.asarray(fold_probabilities)
        beams_by_fold = np.asarray(fold_beams)
        ensemble = by_fold.mean(axis=0)
        np.testing.assert_allclose(ensemble, beams_by_fold.mean(axis=(0, 2)), atol=1e-6, rtol=0)
        np.save(self.output / "test_probabilities_by_fold.npy", by_fold)
        np.save(self.output / "test_beam_probabilities_by_fold.npy", beams_by_fold)
        np.save(self.output / "test_probabilities.npy", ensemble)
        model_metrics = reference_metrics(data["test_model_y"], ensemble)
        project_metrics = current_label_metrics(data["test_current_y"], ensemble)
        predictions = (ensemble[:, 1] >= 0.5).astype(np.int64)
        with (self.output / "test_predictions.csv").open(
            "w", encoding="utf-8-sig", newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "point_id", "reference_model_label", "reference_prediction",
                "probability_thin", "project_label", "project_prediction",
            ])
            writer.writerows([
                point_id, int(model_label), int(prediction), float(probability[1]),
                int(current_label), int(1 - prediction),
            ] for point_id, model_label, prediction, probability, current_label in zip(
                data["test_ids"], data["test_model_y"], predictions, ensemble,
                data["test_current_y"],
            ))
        result = {
            "status": "complete", "finished_at": stamp(), "folds": fold_results,
            "oof_metrics_reference_labels": oof_metrics,
            "test_metrics_reference_labels": model_metrics,
            "test_metrics_project_labels": project_metrics,
            "target_reference_accuracy": 61 / 67,
            "target_reached": model_metrics["correct"] >= 61,
            "intentional_difference": (
                "cubic spline replaces reference linear interpolation"
                if self.options.resampling == RESAMPLING_CUBIC else None
            ),
        }
        atomic_json(self.output / "result.json", result)
        self.update(status="complete", fold=None, epoch=0, result=result)
        lines = [
            "# 六路 Beam 投票参考协议对齐结果", "",
            f"状态：complete；完成时间：{result['finished_at']}", "",
            f"测试正确数：{model_metrics['correct']}/67",
            f"测试 accuracy：{model_metrics['accuracy']:.4%}",
            f"测试 AUC：{model_metrics['auc']:.6f}",
            f"目标 61/67：{'达到' if result['target_reached'] else '未达到'}", "",
            f"重采样：{self.options.resampling}；其余对齐共享全段 max1、双 Beam 融合头、",
            "双 Beam 交叉熵、参考五折、验证 AUC 选 epoch、BF16、fused AdamW、参考学习率和五折概率平均。",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=APP_DIR / "raw_data_relabeled/统一_cls2_thr1.3",
    )
    parser.add_argument("--reference-root", type=Path, default=APP_DIR.parent / "code_reference")
    parser.add_argument(
        "--output-dir", type=Path,
        default=APP_DIR / "experiments/reference_alignment_full_20261027",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument(
        "--resampling", choices=(RESAMPLING_CUBIC, RESAMPLING_REFERENCE_LINEAR),
        default=RESAMPLING_CUBIC,
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.reference_root = args.reference_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.folds = [int(value) for value in args.folds.split(",") if value.strip()]
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(value not in range(1, 6) for value in args.folds):
        parser.error("--folds must contain unique values from 1 through 5")
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("the aligned protocol requires a BF16-capable CUDA GPU")
    return args


def main() -> None:
    options = parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    run = AlignedRun(options)
    try:
        run.run()
    except BaseException:
        run.update(status="failed", error=traceback.format_exc(), failed_at=stamp())
        raise


if __name__ == "__main__":
    main()
