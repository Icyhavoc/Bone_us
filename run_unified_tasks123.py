"""Validation-only ViT sweeps with the requested task-1 label-1 stop gate."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import run_vit_experiments as trainer  # Initialize the shared spline runtime first.
import torch
from dyn_cli import dyn_config_from_args
from emd_pipeline import DepthMapper, read_label_scheme, write_json
from experiment_setup import region_experiments
from vit_model import RegionChannelCrossVoteViT, RegionChannelViT, ViTConfig

APP_DIR = Path(__file__).resolve().parent
SOURCE_FILES = ("run_unified_tasks123.py", "run_vit_experiments.py", "vit_model.py",
                "vit_pipeline.py", "emd_pipeline.py", "dyn_cli.py",
                "experiment_setup.py", "learning_rate_schedule.py", "vit_hyperparameters.py")


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 11:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 0.5))
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass  # Retain only this failed write's private temporary file.


def candidates(reference_dir):
    unique = {}
    for source in sorted(reference_dir.glob("*top15.json")):
        data = json.loads(source.read_text("utf-8"))
        for row in data["configurations"]:
            signature = json.dumps(row["parameters"], sort_keys=True)
            entry = unique.setdefault(signature, {"id": row["id"], "parameters": row["parameters"], "references": []})
            entry["references"].append({"file": source.name, "method": data["method"], "id": row["id"]})
    if not unique:
        raise ValueError("no reference hyperparameters found")
    return sorted(unique.values(), key=lambda row: row["id"])


def stop_for_label1(metrics, limit=0.4):
    fn, tp = int(metrics["fn"]), int(metrics["tp"])
    if fn + tp == 0:
        raise ValueError("label=1 is absent from the evaluation split")
    return {"split": "test", "label": 1, "wrong": fn, "total": fn + tp,
            "error_rate": fn / (fn + tp), "limit": limit,
            "stop": fn > limit * (fn + tp), "rule": "strictly greater than limit"}


class Batch:
    source_files = SOURCE_FILES
    channel_mode = 3
    model_variant = "concat"

    def manifest_extra(self):
        return {}

    def __init__(self, options):
        self.options = options
        self.root = options.output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = options.data_dir.resolve()
        self.scheme = read_label_scheme(self.data_dir)
        if not self.scheme or self.scheme["tag"] != "cls2_thr1.3_unified":
            raise ValueError("this batch requires the unified cls2_thr1.3 dataset")
        pool = candidates(options.reference_dir.resolve())
        self.pool = pool if options.limit is None else pool[:options.limit]
        self.progress = {"status": "preparing", "pid": os.getpid(), "started_at": stamp(),
                         "stage": "preparing", "completed": 0, "total": len(self.pool),
                         "epoch": 0, "epoch_limit": options.epochs, "results": []}
        self.cache = {}
        self.test_cache = {}
        self.rows = []
        manifest = {
            "dataset": str(self.data_dir), "candidate_count": len(self.pool), "candidates": self.pool,
            "training": {"epochs": options.epochs, "patience": 12, "seed": 2023,
                         "decision_threshold": 0.5, "reward": False, "max_samples": options.max_samples,
                         "lr_schedule": "warmup_cosine", "min_learning_rate": 5e-6},
            "selection": ["validation accuracy descending", "validation loss ascending", "configuration ID ascending"],
            "epoch_selection": "validation loss minimum", "test_during_sweep": False,
            "stop_gate": {"task": 1, "split": "test", "label": 1, "error_limit": options.label1_error_limit},
            "source_sha256": {name: hashlib.sha256((APP_DIR / name).read_bytes()).hexdigest() for name in self.source_files},
            "dataset_sha256": {str(p.relative_to(self.data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for split in ("train", "val", "test")
                               for p in sorted((self.data_dir / split).iterdir()) if p.is_file()},
            "reference_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(options.reference_dir.resolve().glob("*top15.json"))},
        }
        manifest.update(self.manifest_extra())
        old = self.root / "manifest.json"
        if old.exists() and json.loads(old.read_text("utf-8")) != manifest:
            raise ValueError("batch protocol or source changed; choose another output directory")
        atomic_json(old, manifest)
        snapshots = self.root / "sources"
        snapshots.mkdir(exist_ok=True)
        for name in self.source_files:
            shutil.copy2(APP_DIR / name, snapshots / name)
        original = sys.argv
        try:
            sys.argv = ["run_vit_experiments.py", "--data-dir", str(self.data_dir), "--no-reward",
                        "--defer-test", "--device", options.device, "--channel-mode", "3",
                        "--hyperparameter-preset", "legacy", "--epochs", str(options.epochs)]
            if options.max_samples is not None:
                sys.argv += ["--max-samples", str(options.max_samples)]
            self.base = trainer._parse_args()
        finally:
            sys.argv = original
        self.device = trainer._device(options.device)
        torch.set_num_threads(4)
        self.update()

    def update(self, **fields):
        self.progress.update(fields)
        self.progress["updated_at"] = stamp()
        try:
            atomic_json(self.root / "progress.json", self.progress)
        except PermissionError as exc:
            # Progress is advisory: an occupied status file must not stop a fit.
            self.progress["progress_write_warning"] = str(exc)
            print(f"Progress update delayed; training continues: {exc}", file=sys.stderr, flush=True)
            try:
                atomic_json(self.root / "progress_fallback.json", self.progress)
            except PermissionError as fallback_exc:
                print(f"Fallback progress also occupied: {fallback_exc}", file=sys.stderr, flush=True)

    def prepared(self, region, split):
        target = self.test_cache if split == "test" else self.cache
        key = region, self.channel_mode, split
        if key not in target:
            _, regions = region_experiments(region)[0]
            target[key] = trainer._load_prepared_split(
                self.data_dir, split, form="max1", regions=regions, channel_mode=self.channel_mode,
                mapper=DepthMapper(), target_length=512, tukey_alpha=0.3,
                dynamic_config=dyn_config_from_args(self.base), max_samples=self.options.max_samples)
        return target[key]

    def fit(self, stage, region, candidate, directory):
        metrics_path = directory / "metrics.json"
        if metrics_path.exists() and (directory / "vit_model.pt").exists():
            metrics = json.loads(metrics_path.read_text("utf-8"))
        else:
            args = copy.copy(self.base)
            args.region_set = region
            args.channel_mode = self.channel_mode
            args.model_variant = self.model_variant
            args.search_candidate = candidate
            for key, value in candidate["parameters"].items():
                if key == "label_smoothing":
                    if value != 0:
                        raise ValueError("only the existing unsmoothed CE objective is supported")
                    continue
                setattr(args, {"lr": "learning_rate", "warmup": "warmup_epochs"}.get(key, key), value)
            args.epoch_callback = lambda epoch, best: self.update(epoch=epoch["epoch"],
                current_val_accuracy=epoch["val_accuracy"], current_val_loss=epoch["val_loss"], best_epoch=best)
            prepared = {split: self.prepared(region, split) for split in ("train", "val")}
            _, regions = region_experiments(region)[0]
            started = time.monotonic()
            metrics = trainer.run_one(self.data_dir, directory, "max1", region, regions,
                                      args, 2, self.scheme,
                                      None if self.model_variant == "cross_vote" else "fixed0p5",
                                      prepared_splits=prepared)
            atomic_json(directory / "runtime.json", {"seconds": time.monotonic() - started})
            torch.cuda.empty_cache()
        return {"candidate": candidate, "directory": str(directory), "region": region,
                "channel_mode": self.channel_mode, "model_variant": self.model_variant,
                "val_accuracy": metrics["metrics"]["val"]["accuracy"],
                "val_loss": metrics["best_val_loss"], "best_epoch": metrics["best_epoch"],
                "trained_epochs": metrics["trained_epochs"]}

    def sweep(self, stage, region, reuse=None):
        rows = []
        self.update(stage=stage, completed=0, total=len(self.pool), epoch=0,
                    best_candidate=None, best_val_accuracy=None)
        for position, candidate in enumerate(self.pool):
            self.update(current_candidate=candidate["id"], epoch=0, current_val_accuracy=None)
            if reuse is not None and candidate["id"] == reuse["candidate"]["id"]:
                row = reuse
            else:
                row = self.fit(stage, region, candidate, self.root / stage / candidate["id"])
            rows.append(row)
            rows.sort(key=lambda r: (-r["val_accuracy"], r["val_loss"], r["candidate"]["id"]))
            atomic_json(self.root / stage / "sweep_results.json", rows)
            self.update(completed=position + 1, best_candidate=rows[0]["candidate"]["id"],
                        best_val_accuracy=rows[0]["val_accuracy"])
            print(f"SWEEP {stage}: {position + 1}/{len(self.pool)} best={rows[0]['candidate']['id']} val={rows[0]['val_accuracy']:.4f}", flush=True)
        atomic_json(self.root / stage / "selection.json", {"selected_before_test_at": stamp(), "winner": rows[0]})
        return rows[0]

    def evaluate(self, task, row):
        directory = Path(row["directory"])
        metrics = json.loads((directory / "metrics.json").read_text("utf-8"))
        if "test" not in metrics["metrics"]:
            self.update(stage=f"task{task}_test_evaluation", current_candidate=row["candidate"]["id"])
            inputs, labels, records, audit = self.prepared(row["region"], "test")
            checkpoint = torch.load(directory / "vit_model.pt", map_location="cpu", weights_only=False)
            vote = row.get("model_variant", "concat") == "cross_vote"
            model_class = RegionChannelCrossVoteViT if vote else RegionChannelViT
            model = model_class(ViTConfig(**checkpoint["config"])).to(self.device)
            model.load_state_dict(checkpoint["model"])
            criterion = torch.nn.NLLLoss() if vote else torch.nn.CrossEntropyLoss()
            probabilities, loss = trainer._evaluate(model, inputs, labels, criterion,
                                                     row["candidate"]["parameters"]["batch_size"], self.device)
            if vote:
                decision = trainer.classification_metrics(labels, probabilities, 2, binary_decision="argmax")
                decision["decision_mode"] = "vote_argmax"
                decision["weighted_balanced_accuracy"] = (decision["specificity"] + decision["sensitivity"]) / 2
                metrics["metrics"]["test"] = decision
                branches = trainer._cross_vote_branch_probabilities(
                    model, inputs, row["candidate"]["parameters"]["batch_size"], self.device)
                np.save(directory / "branch_probabilities_test.npy", branches)
                np.save(directory / "vote_scores_test.npy", branches.sum(axis=1))
            else:
                metrics["metrics"]["test"] = trainer._binary_metrics(labels, probabilities, 0.5, None)
            metrics["test_loss"] = loss
            metrics["split_counts"]["test"] = len(labels)
            metrics["class_counts"]["test"] = {str(i): int(np.sum(labels == i)) for i in (0, 1)}
            np.save(directory / "labels_test.npy", labels)
            np.save(directory / "probabilities_test.npy", probabilities)
            write_json(directory / "samples_test.json", records)
            write_json(directory / "input_audit_test.json", audit)
            write_json(directory / "metrics.json", metrics)
            prediction = (probabilities.argmax(1) if vote else (probabilities[:, 1] >= 0.5).astype(int))
            with (directory / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "label", "prediction", "probability_label1"])
                writer.writerows((record["sample_id"], int(label), int(pred), float(prob[1]))
                                for record, label, pred, prob in zip(records, labels, prediction, probabilities))
            del model
            torch.cuda.empty_cache()
        labels = np.load(directory / "labels_test.npy")
        probabilities = np.load(directory / "probabilities_test.npy")
        prediction = (probabilities.argmax(1) if row.get("model_variant") == "cross_vote"
                      else (probabilities[:, 1] >= 0.5).astype(int))
        matrix = np.bincount(labels * 2 + prediction, minlength=4).reshape(2, 2)
        metrics["metrics"]["test"]["confusion_matrix"] = matrix.tolist()
        write_json(directory / "metrics.json", metrics)
        result = {"task": task, **row, "test": metrics["metrics"]["test"]}
        self.rows.append(result)
        atomic_json(self.root / "results.json", self.rows)
        self.update(results=self.rows)
        return result

    def report(self, status, gate):
        atomic_json(self.root / "gate.json", gate)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, len(self.rows), figsize=(5 * len(self.rows), 4), squeeze=False)
        for axis, row in zip(axes[0], self.rows):
            matrix = np.asarray(row["test"]["confusion_matrix"])
            axis.imshow(matrix, cmap="Blues", vmin=0)
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18)
            axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label",
                     title=f"Task {row['task']} / {row['candidate']['id']}\nAccuracy {row['test']['accuracy']:.2%}")
        figure.tight_layout()
        figure.savefig(self.root / "confusion_matrices.png", dpi=180)
        plt.close(figure)
        lines = ["# 统一数据集任务 1–3", "", f"状态：{status}",
                 f"任务1 label=1 错分：{gate['wrong']}/{gate['total']} = {gate['error_rate']:.2%}；停止阈值 > {gate['limit']:.0%}。", "",
                 "|任务|配置|验证accuracy|测试accuracy|薄骨召回率|厚骨召回率|AUC|", "|---|---|---:|---:|---:|---:|---:|"]
        for row in self.rows:
            m = row["test"]
            lines.append(f"|{row['task']}|{row['candidate']['id']}|{row['val_accuracy']:.2%}|{m['accuracy']:.2%}|{m['specificity']:.2%}|{m['sensitivity']:.2%}|{m['auc']:.4f}|")
        lines += ["", "全部候选仅按验证 accuracy、验证 loss、配置 ID 排序；epoch 按验证 loss 选择。",
                  "统一数据集 161/40/67 点，max1、channel3 独立双 Beam 选帧、reward off、阈值 0.5、seed2023。",
                  "原始参数、候选列表、源码与数据哈希见 manifest.json / sources/；逐轮历史及逐样本概率见各配置目录。", ""]
        for row in self.rows:
            lines += [f"## 任务 {row['task']} 参数", "", "```json", json.dumps(row["candidate"]["parameters"], indent=2), "```", ""]
        (self.root / "REPORT.md").write_text("\n".join(lines), "utf-8")
        self.update(status=status, stage=status, gate=gate, finished_at=stamp())

    def run(self):
        self.update(status="running")
        first = self.evaluate(1, self.sweep("task1_baseline_search", "dyn_envelope"))
        gate = stop_for_label1(first["test"], self.options.label1_error_limit)
        atomic_json(self.root / "gate.json", gate)
        if gate["stop"]:
            self.report("stopped_by_label1_guard", gate)
            return
        baseline = first["candidate"]
        self.update(stage="task2_six_channel_baseline_parameters", completed=0, total=1, epoch=0,
                    current_candidate=baseline["id"])
        second = self.fit("task2", "dyn_envelope_full", baseline, self.root / "task2_baseline_parameters")
        self.evaluate(2, second)
        self.evaluate(3, self.sweep("task3_six_channel_search", "dyn_envelope_full", reuse=second))
        self.report("complete", gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=APP_DIR / "raw_data_relabeled/统一_cls2_thr1.3")
    parser.add_argument("--reference-dir", type=Path, default=APP_DIR.parent / "ViT_reference/top15_hyperparameters_test_accuracy_20261024")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "experiments/unified_tasks123_20261025")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--label1-error-limit", type=float, default=0.4)
    args = parser.parse_args()
    batch = Batch(args)
    try:
        batch.run()
    except BaseException:
        batch.update(status="failed", stage="failed", error=traceback.format_exc(), failed_at=stamp())
        raise


if __name__ == "__main__":
    main()
