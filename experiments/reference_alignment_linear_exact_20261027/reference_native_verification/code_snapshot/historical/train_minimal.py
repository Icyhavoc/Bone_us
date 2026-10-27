# -*- coding: utf-8 -*-
"""按师兄 CFG / Signalset / Trainer 结构整理的主体、尾部联合 ViT。"""
import argparse
import io
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models.vit_1d_cls import ViT
from prepare_data import ROOT, audit_dataset, make_input, input_metadata, save_json, sha256
from configs import CFG, INPUT_INDEX, FIXED_SEQ_LEN, get_cfg
from prepare_data import read_dataset_info, input_window_metadata


# =========================================================
# 0. 可复现实验：固定随机种子
# =========================================================
def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# =========================================================
# 1. CFG：在 configs.py 修改 INPUT_INDEX；每种方案使用独立 CFG class
# =========================================================
# CFGMeanStd / CFGRaw50 / CFGMaxFrame / CFGTop3Mean
# 命令行 --input-index 优先于文件内默认值，所有默认参数随选中的类一起切换。


# =========================================================
# 2. 数据集：一个点位是一条样本，主体与尾部保持配对
# =========================================================
class Signalset(Dataset):
    def __init__(self, data, scaling=None):
        self.label = torch.from_numpy(data["label"])
        self.point_ids = data["point_ids"]
        self.indices = data["indices"]
        self.input_name = data["input_name"]
        if scaling is not None:
            raise ValueError("当前流程不使用RMS缩放；旧缩放checkpoint需使用其历史代码读取。")
        # 保留checkpoint字段为None，供当前训练与搜索入口统一读取；不拟合任何缩放参数。
        self.scaling = None
        self.signal = [torch.from_numpy(data[region]) for region in ("main", "tail")]
        if not all(torch.isfinite(value).all() for value in self.signal):
            raise ValueError("模型输入包含 NaN 或 Inf")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {"main": self.signal[0][idx], "tail": self.signal[1][idx],
                "label": self.label[idx]}


# =========================================================
# 3. 构建模型
# =========================================================
def fetch_model(args):
    return ViT(seq_len=args.seq_len, patch_size=args.patch_size,
               num_classes=args.num_classes, dim=args.dim, depth=args.depth,
               heads=args.heads, mlp_dim=args.mlp_dim, channels=args.channels,
               dropout=args.dropout, frames=args.frames,
               main_seq_len=args.main_seq_len, tail_seq_len=args.tail_seq_len,
               split_beams=getattr(args, "split_beams", False))


# =========================================================
# 4. 指标与阈值：1 表示厚度低于数据集分界点；概率阈值仅用验证集选择
# =========================================================
def compute_metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(2), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(2), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=(precision + recall) != 0)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "sensitivity": float(recall[1]),
        "specificity": float(recall[0]),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "correct": int(np.sum(labels == predictions)), "n": len(labels),
        "threshold": float(threshold), "confusion_matrix": cm.tolist(),
    }


def select_threshold(labels, probabilities):
    unique = np.unique(probabilities)
    candidates = np.unique(np.r_[0., .5, (unique[:-1] + unique[1:]) / 2,
                                  np.nextafter(1., 2.)])
    def key(threshold):
        metrics = compute_metrics(labels, probabilities, threshold)
        return (metrics["accuracy"], metrics["balanced_accuracy"],
                -abs(threshold - .5), -threshold)
    return float(max(candidates, key=key))


# =========================================================
# 5. 训练器：两个分支共同反向传播，只有一个分类头和优化器
# =========================================================
class Trainer:
    def __init__(self, args, train_data, val_data, seed, output):
        set_seed(seed)
        self.args = args
        self.seed = seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(args.device)
        self.model = fetch_model(args).to(self.device)
        self.crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.train_set = Signalset(train_data)
        self.val_set = Signalset(val_data, self.train_set.scaling)
        if len(np.unique(train_data["label"])) != 2 or len(np.unique(val_data["label"])) != 2:
            raise ValueError("训练集、验证集均须包含两个类别以按验证 AUC 选择权重")
        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size,
                                       shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_set, batch_size=64, shuffle=False, num_workers=0)

    def _forward(self, batch):
        return self.model(batch["main"].to(self.device), batch["tail"].to(self.device))

    def _train_one_epoch(self):
        self.model.train()
        total = 0.
        for batch in self.train_loader:
            labels = batch["label"].to(self.device)
            self.optim.zero_grad(set_to_none=True)
            loss = self.crit(self._forward(batch), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失不是有限值")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optim.step()
            total += loss.item() * len(labels)
        return total / len(self.train_set)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        probabilities, labels = [], []
        loss_sum = 0.
        for batch in loader:
            logits = self._forward(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("预测输出不是有限值")
            loss_sum += self.crit(logits, batch["label"].to(self.device)).item() * len(logits)
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
        return np.concatenate(probabilities), np.concatenate(labels), loss_sum / len(loader.dataset)

    def train(self):
        best_auc, stale, history = -1., 0, []
        for epoch in range(1, self.args.epochs + 1):
            start = time.perf_counter()
            if epoch <= self.args.warmup:
                factor = epoch / max(1, self.args.warmup)
            else:
                progress = (epoch - self.args.warmup) / max(1, self.args.epochs - self.args.warmup)
                factor = .5 * (1 + np.cos(np.pi * progress))
            for group in self.optim.param_groups:
                group["lr"] = self.args.lr * factor
            loss = self._train_one_epoch()
            probabilities, labels, val_loss = self.evaluate(self.val_loader)
            metrics = compute_metrics(labels, probabilities)
            improved = metrics["roc_auc"] > best_auc + 1e-8
            if improved:
                best_auc, stale = metrics["roc_auc"], 0
                torch.save({"model": self.model.state_dict(), "scaling": self.train_set.scaling,
                            "seed": self.seed, "epoch": epoch, "validation_auc": best_auc,
                            "config": serializable_args(self.args)}, self.output / "best.pt")
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": loss, "validation_loss": val_loss,
                            "validation_auc": metrics["roc_auc"],
                            "validation_accuracy_05": metrics["accuracy"],
                            "lr": self.optim.param_groups[0]["lr"]})
            save_json(self.output / "history.json", history)
            print(f"[Seed {self.seed} | Epoch {epoch:03d}] loss={loss:.4f} | "
                  f"val_auc={metrics['roc_auc']:.4f} | val_acc={metrics['accuracy']:.4f} | "
                  f"time={time.perf_counter() - start:.1f}s{' *' if improved else ''}", flush=True)
            if stale >= self.args.patience:
                break
        checkpoint = torch.load(self.output / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        probabilities, labels, _ = self.evaluate(self.val_loader)
        np.save(self.output / "validation_probabilities.npy", probabilities)
        save_json(self.output / "summary.json", {
            "seed": self.seed, "best_epoch": checkpoint["epoch"], "epochs_run": len(history),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "validation_at_05": compute_metrics(labels, probabilities),
        })
        return probabilities


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def export_predictions(path, data, probabilities, threshold):
    import csv
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_index", "point_id", "true_label", "probability_below_boundary",
                         "predicted_label", "correct", "boundary_mm"])
        predictions = (probabilities >= threshold).astype(int)
        writer.writerows(zip(data["indices"], data["point_ids"], data["label"],
                             probabilities, predictions, predictions == data["label"],
                             np.full(len(predictions), data["boundary_mm"])))


def smoke_test(trainer):
    """真实训练信号执行一步更新；检查两个分支均收到梯度及权重往返。"""
    batch = next(iter(DataLoader(trainer.train_set, batch_size=4)))
    trainer.model.train()
    trainer.optim.zero_grad(set_to_none=True)
    loss = trainer.crit(trainer._forward(batch), batch["label"].to(trainer.device))
    if not torch.isfinite(loss):
        raise AssertionError("smoke loss is not finite")
    loss.backward()
    norms = []
    for branch in trainer.model.branches:
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("branch gradient is missing or non-finite")
        norms.append(float(sum(g.detach().norm().item() for g in gradients)))
    if min(norms) <= 0:
        raise AssertionError("both branches must receive gradients")
    trainer.optim.step()
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer._forward(batch)
    buffer = io.BytesIO()
    torch.save({"model": trainer.model.state_dict(), "scaling": trainer.train_set.scaling}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True, map_location=trainer.device)
    restored = fetch_model(trainer.args).to(trainer.device)
    restored.load_state_dict(checkpoint["model"])
    restored.eval()
    with torch.no_grad():
        actual = restored(batch["main"].to(trainer.device), batch["tail"].to(trainer.device))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probabilities, labels, val_loss = trainer.evaluate(trainer.val_loader)
    if probabilities.shape != labels.shape or not np.isfinite(val_loss):
        raise AssertionError("validation loader failed")
    record = {"status": "passed", "kind": "smoke_test_only", "loss": loss.item(),
              "input_index": trainer.args.input_index, "input_name": trainer.args.input_name,
              "branch_gradient_norms": norms, "checkpoint_roundtrip": "exact",
              "parameters": sum(p.numel() for p in restored.parameters()),
              "train_points": len(trainer.train_set), "validation_points": len(labels),
              "test_inference_performed": False}
    save_json(trainer.output / "smoke_test.json", record)
    print("[SMOKE] PASS: paired input, both branch gradients, checkpoint, validation loader")


# =========================================================
# 6. 参数与主函数
# =========================================================
def get_args(argv=None):
    # 先解析方案索引，再用该方案的独立CFG生成其余参数默认值。
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--input-index", type=int, choices=range(4), default=INPUT_INDEX)
    selected, _ = selector.parse_known_args(argv)
    cfg = get_cfg(selected.input_index)
    parser = argparse.ArgumentParser("主体/尾部联合 Transformer 二分类", parents=[selector])
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--output", type=Path, help="新的输出目录；已有目录不会被覆盖")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.seeds))
    for name in ("seq_len", "patch_size", "dim", "depth", "heads", "mlp_dim", "epochs",
                 "patience", "batch_size", "warmup"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(cfg, name))
    for name in ("dropout", "lr", "weight_decay", "label_smoothing"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(cfg, name))
    parser.add_argument("--skip-test", action="store_true", help="训练和验证后结束")
    parser.add_argument("--smoke-test", action="store_true", help="只检查数据和一步联合更新")
    parser.add_argument("--rank-metric", choices=("peak",), default=cfg.rank_metric)
    args = parser.parse_args(argv)
    args.input_name = cfg.input_name
    args.cfg_class = cfg.__name__
    args.channels, args.frames, args.num_classes = cfg.channels, cfg.frames, cfg.num_classes
    args.split_beams = getattr(cfg, "split_beams", False)
    args.model_architecture = "four_region_beam_vit" if args.split_beams else "two_region_vit"
    args.branch_names = (["main_beam1", "main_beam2", "tail_beam1", "tail_beam2"]
                         if args.split_beams else ["main", "tail"])
    args.main_seq_len = args.tail_seq_len = FIXED_SEQ_LEN
    args.input_window = input_window_metadata()
    args.input_scaling = "none"
    if args.seq_len != FIXED_SEQ_LEN:
        parser.error(f"所有方案固定 seq-len={FIXED_SEQ_LEN}")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.dataset_info = read_dataset_info(args.dataset_root)
    args.boundary_mm = args.dataset_info["boundary_mm"]
    if min(args.patch_size, args.dim, args.depth, args.heads) < 1 or args.dim % 2 or args.dim % args.heads:
        parser.error("模型维度须为正偶数并能被 heads 整除")
    expected_shape = ((4, 1), (2, 50), (2, 1), (2, 1))[args.input_index]
    if (args.channels, args.frames) != expected_shape or args.num_classes != 2:
        parser.error("CFG 的 channels/frames/num_classes 与该输入方案不匹配")
    if min(args.epochs, args.patience, args.batch_size, args.mlp_dim) < 1 or args.warmup < 0:
        parser.error("epoch、patience、batch-size、mlp-dim 须为正数，warmup 不能为负")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1:
        parser.error("学习率、权重衰减或 dropout/label-smoothing 范围错误")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("随机种子须唯一且处于 [0, 2**32) 范围")
    if args.output is None:
        parent = "verification" if args.smoke_test else "results"
        args.output = ROOT / parent / args.dataset_info["dataset_tag"] / args.input_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return args


def main():
    args = get_args()
    torch.set_num_threads(4)
    print(f"[INPUT {args.input_index}] {args.input_name} / {args.cfg_class}", flush=True)
    summary = audit_dataset(args.dataset_root, require_source_signal=args.input_index in (2, 3))
    print(f"[DATASET] {args.dataset_root} | boundary={args.boundary_mm:g} mm", flush=True)
    # 每次运行使用新输出目录，避免不同设置的权重/指标混用。
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", serializable_args(args))
    save_json(args.output / "data_audit.json", summary)
    save_json(args.output / "dataset_info.json", args.dataset_info)
    save_json(args.output / "input_code_manifest.json", {
        "dataset_sha256": {p.relative_to(args.dataset_root).as_posix(): sha256(p)
                           for p in sorted(args.dataset_root.rglob("*")) if p.is_file()},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                        for p in [Path(__file__), ROOT / "configs.py", ROOT / "prepare_data.py", ROOT / "models/vit_1d_cls.py"]},
    })
    train_data = make_input(args.dataset_root, "train", args.seq_len, args.input_index, args.rank_metric)
    val_data = make_input(args.dataset_root, "validation", args.seq_len, args.input_index, args.rank_metric)
    save_json(args.output / "train_input.json", input_metadata(train_data))
    save_json(args.output / "validation_input.json", input_metadata(val_data))
    if args.smoke_test:
        smoke_test(Trainer(args, train_data, val_data, args.seeds[0], args.output / "smoke"))
        return
    val_predictions = []
    for seed in args.seeds:
        trainer = Trainer(args, train_data, val_data, seed, args.output / f"seed_{seed}")
        val_predictions.append(trainer.train())
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probabilities = np.mean(val_predictions, axis=0)
    threshold = select_threshold(val_data["label"], probabilities)
    checkpoints = {f"seed_{seed}/best.pt": sha256(args.output / f"seed_{seed}/best.pt")
                   for seed in args.seeds}
    frozen = {"method": "equal mean probability of jointly trained models",
              "input_index": args.input_index, "input_name": args.input_name,
              "seeds": args.seeds, "threshold": threshold, "selected_on": "validation",
              "checkpoints": checkpoints, "labels": args.dataset_info["model_labels"],
              "dataset_info": args.dataset_info,
              "validation": compute_metrics(val_data["label"], probabilities, threshold)}
    save_json(args.output / "frozen_selection.json", frozen)
    export_predictions(args.output / "validation_predictions.csv", val_data, probabilities, threshold)
    if args.skip_test:
        print("[DONE] 训练与验证已完成，未执行测试集推理。", args.output)
        return
    # 最佳权重、模型成员及分类阈值先冻结，之后才读取测试输入进行推理。
    test_data = make_input(args.dataset_root, "test", args.seq_len, args.input_index, args.rank_metric)
    save_json(args.output / "test_input.json", input_metadata(test_data))
    test_predictions = []
    for relative, expected_hash in checkpoints.items():
        checkpoint_path = args.output / relative
        if sha256(checkpoint_path) != expected_hash:
            raise RuntimeError("冻结后权重发生变化")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        model = fetch_model(args).to(args.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        dataset = Signalset(test_data, checkpoint["scaling"])
        parts = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=64, shuffle=False):
                logits = model(batch["main"].to(args.device), batch["tail"].to(args.device))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("测试预测包含 NaN 或 Inf")
                parts.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
        test_predictions.append(np.concatenate(parts))
        del model
    probabilities = np.mean(test_predictions, axis=0)
    metrics = compute_metrics(test_data["label"], probabilities, threshold)
    export_predictions(args.output / "test_predictions.csv", test_data, probabilities, threshold)
    save_json(args.output / "test_result.json", {
        "metrics": metrics, "selection_sha256": sha256(args.output / "frozen_selection.json"),
        "dataset_info": args.dataset_info,
        "scope": "existing point-level internal holdout; previously inspected; not bone-grouped",
    })
    print(f"[TEST] accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['n']}) | "
          f"AUC={metrics['roc_auc']:.4f}")
    print("[DONE]", args.output)


if __name__ == "__main__":
    main()
