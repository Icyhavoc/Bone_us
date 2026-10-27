"""Run tasks 4–7 with the frozen task-1 baseline and the common 52 candidates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

from run_unified_tasks123 import APP_DIR, SOURCE_FILES, Batch, atomic_json, stamp


class RemainingBatch(Batch):
    source_files = (*SOURCE_FILES, "run_unified_tasks4567.py")

    def __init__(self, options):
        self.prior_root = options.baseline_dir.resolve()
        prior_status = json.loads((self.prior_root / "progress.json").read_text("utf-8"))
        if prior_status["status"] != "complete":
            raise ValueError("tasks 1–3 must be complete before starting tasks 4–7")
        self.prior_gate = json.loads((self.prior_root / "gate.json").read_text("utf-8"))
        if self.prior_gate["stop"]:
            raise ValueError("task-1 label-1 error gate failed; remaining tasks cannot start")
        self.prior_rows = json.loads((self.prior_root / "results.json").read_text("utf-8"))
        self.baseline = copy.deepcopy(next(row["candidate"] for row in self.prior_rows if row["task"] == 1))
        self.prior_manifest = json.loads((self.prior_root / "manifest.json").read_text("utf-8"))
        # Preserve the exact training/model/preprocessing used in tasks 1–3.
        for name, expected in self.prior_manifest["source_sha256"].items():
            if name == "run_unified_tasks123.py":
                continue  # The batch driver now routes channel4 and cross-vote too.
            if hashlib.sha256((APP_DIR / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"training source differs from tasks 1–3: {name}")
        super().__init__(options)
        manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        if manifest["dataset_sha256"] != self.prior_manifest["dataset_sha256"]:
            raise ValueError("unified dataset differs from tasks 1–3")
        if manifest["reference_sha256"] != self.prior_manifest["reference_sha256"]:
            raise ValueError("reference candidate files differ from tasks 1–3")
        if options.limit is None and self.pool != self.prior_manifest["candidates"]:
            raise ValueError("candidate pool differs from tasks 1–3")

    def manifest_extra(self):
        return {
            "tasks": [4, 5, 6, 7],
            "baseline_candidate": self.baseline,
            "baseline_results_sha256": hashlib.sha256((self.prior_root / "results.json").read_bytes()).hexdigest(),
            "prior_manifest_sha256": hashlib.sha256((self.prior_root / "manifest.json").read_bytes()).hexdigest(),
            "stop_gate": {"task": 1, "already_evaluated": self.prior_gate, "new_stop_rules": None},
            "methods": {
                "tasks4_5": {"form": "max1", "region": "dyn_envelope", "channel_mode": 4,
                             "model_variant": "concat", "reward": False, "threshold": 0.5},
                "tasks6_7": {"form": "max1", "region": "dyn_envelope", "channel_mode": 3,
                             "model_variant": "cross_vote", "branches": 4,
                             "loss": "mean-probability NLL", "decision": "vote_argmax"},
            },
        }

    def profile(self, channel, variant):
        self.channel_mode, self.model_variant = channel, variant

    def fixed(self, task, channel, variant):
        self.profile(channel, variant)
        self.update(stage=f"task{task}_baseline_parameters", completed=0, total=1, epoch=0,
                    current_candidate=self.baseline["id"], best_candidate=None, best_val_accuracy=None)
        row = self.fit(f"task{task}", "dyn_envelope", self.baseline,
                       self.root / f"task{task}_baseline_parameters")
        self.evaluate(task, row)
        self.report_remaining("running")
        return row

    def report_remaining(self, status):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        rows = sorted([*self.prior_rows, *self.rows], key=lambda row: row["task"])
        atomic_json(self.root / "results_all7.json", rows)
        lines = ["# 统一数据集任务 1–7", "", f"状态：{status}",
                 "任务1–3复用已完成结果；任务4/6固定任务1选出的完整超参，任务5/7各搜索同一份52套候选。", "",
                 "|任务|配置|验证accuracy|测试accuracy|薄骨召回率|厚骨召回率|厚骨错分率|AUC|",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        names = {1: "baseline best", 2: "6 branches / baseline HP", 3: "6 branches / best HP",
                 4: "channel4 / baseline HP", 5: "channel4 / best HP",
                 6: "cross-vote / baseline HP", 7: "cross-vote / best HP"}
        figure, axes = plt.subplots(2, 4, figsize=(20, 9))
        for axis in axes.ravel():
            axis.set_visible(False)
        for axis, row in zip(axes.ravel(), rows):
            m = row["test"]
            error = m["fn"] / (m["fn"] + m["tp"])
            lines.append(f"|{row['task']}|{row['candidate']['id']}|{row['val_accuracy']:.2%}|{m['accuracy']:.2%}|{m['specificity']:.2%}|{m['sensitivity']:.2%}|{error:.2%}|{m['auc']:.4f}|")
            matrix = np.asarray(m["confusion_matrix"])
            axis.set_visible(True)
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=41)
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=17)
            axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label",
                     title=f"Task {row['task']}: {names[row['task']]}\n{row['candidate']['id']} / acc {m['accuracy']:.2%}")
        figure.tight_layout()
        figure.savefig(self.root / "confusion_matrices_all7.png", dpi=180)
        plt.close(figure)
        lines += ["", "label0=薄骨，label1=厚骨。候选按验证accuracy降序、val loss升序、ID升序选优；epoch按val loss选择。",
                  "任务4/5的channel4保留双Top10交集和旧相似度回退；任务6/7使用独立双Beam max1和主体/尾部四分支投票。",
                  "仅对固定参数对照和验证选优后的模型评估test。没有对任务4–7新增40%停止条件。", ""]
        for row in rows:
            lines += [f"## 任务 {row['task']} 参数", "", "```json", json.dumps(row["candidate"]["parameters"], indent=2), "```", ""]
        (self.root / "REPORT.md").write_text("\n".join(lines), "utf-8")
        if status == "complete":
            self.update(status="complete", stage="complete", finished_at=stamp(), gate=self.prior_gate)

    def run(self):
        self.update(status="running", baseline_candidate=self.baseline["id"], gate=self.prior_gate)
        channel4_fixed = self.fixed(4, 4, "concat")
        cross_vote_fixed = self.fixed(6, 3, "cross_vote")
        self.profile(4, "concat")
        self.evaluate(5, self.sweep("task5_channel4_search", "dyn_envelope", reuse=channel4_fixed))
        self.report_remaining("running")
        self.profile(3, "cross_vote")
        self.evaluate(7, self.sweep("task7_cross_vote_search", "dyn_envelope", reuse=cross_vote_fixed))
        self.report_remaining("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=APP_DIR / "raw_data_relabeled/统一_cls2_thr1.3")
    parser.add_argument("--reference-dir", type=Path, default=APP_DIR.parent / "ViT_reference/top15_hyperparameters_test_accuracy_20261024")
    parser.add_argument("--baseline-dir", type=Path, default=APP_DIR / "experiments/unified_tasks123_20261025")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "experiments/unified_tasks4567_20261026")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    args.label1_error_limit = 0.4  # Existing task-1 gate is recorded, not reapplied.
    batch = RemainingBatch(args)
    try:
        batch.run()
    except BaseException:
        batch.update(status="failed", stage="failed", error=traceback.format_exc(), failed_at=stamp())
        raise


if __name__ == "__main__":
    main()
