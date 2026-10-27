# -*- coding: utf-8 -*-
"""直接读取本文件夹内的最新分割信号，生成联合模型输入。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal.windows import tukey

from configs import INPUT_INDEX, FIXED_SEQ_LEN, TUKEY_ALPHA, get_cfg

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")

def input_resampling_metadata():
    return {"method": "time_domain_linear", "target_length": FIXED_SEQ_LEN,
            "axis": -1, "align_endpoints": True, "antialias_filter": False,
            "stage": "after_selection_before_tukey"}

def resample_time_domain(signal, num=FIXED_SEQ_LEN):
    """沿最后采样轴做时域线性插值；首尾对齐，不跨帧/Beam，不额外滤波。"""
    signal = np.asarray(signal)
    if signal.ndim == 0 or signal.shape[-1] < 1:
        raise ValueError("信号须有非空的采样点轴")
    if not isinstance(num, (int, np.integer)) or isinstance(num, (bool, np.bool_)) or num < 1:
        raise ValueError("目标采样点数须为正整数")
    length = signal.shape[-1]
    if length == num:
        return signal.astype(np.float32, copy=True)
    source_positions = np.arange(length, dtype=np.float64)
    target_positions = np.linspace(0, length - 1, num, dtype=np.float64)
    result = np.empty(signal.shape[:-1] + (num,), dtype=np.float32)
    for source_row, target_row in zip(signal.reshape(-1, length), result.reshape(-1, num)):
        target_row[:] = np.interp(target_positions, source_positions, source_row)
    return result

def resample(signal, num, axis=-1):
    if axis != -1:
        raise ValueError("Only the final sample axis is supported")
    return resample_time_domain(signal, num)



def input_window_metadata():
    return {"name": "tukey", "length": FIXED_SEQ_LEN, "alpha": TUKEY_ALPHA,
            "symmetric": True, "axis": -1,
            "stage": "after_resample_before_vit",
            "regions": ["main", "tail"]}


def apply_input_window(signal):
    """每帧、每通道沿最后一轴乘同一对称窗，返回新数组，不修改源数据。"""
    if signal.shape[-1] != FIXED_SEQ_LEN:
        raise ValueError(f"Tukey窗要求重采样后长度为{FIXED_SEQ_LEN}")
    window = tukey(FIXED_SEQ_LEN, alpha=TUKEY_ALPHA, sym=True).astype(np.float32)
    return np.asarray(signal, dtype=np.float32) * window


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset_info(dataset_root):
    """以源元信息确定厚度分界点；概率阈值与厚度分界点是不同概念。"""
    dataset_root = Path(dataset_root).expanduser().resolve()
    metadata_path = dataset_root / "source_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"数据集缺少 source_metadata.json：{dataset_root}")
    metadata = read_json(metadata_path)
    edges = metadata.get("edges")
    if (metadata.get("target_column") != "总厚度" or not isinstance(edges, list)
            or len(edges) != 1 or isinstance(edges[0], bool)
            or not isinstance(edges[0], (int, float)) or not np.isfinite(edges[0])):
        raise ValueError("当前模型为总厚度二分类，source_metadata.json 须含一个有限的分界点 edges=[数值]")
    if metadata.get("interval_rule") != "left_closed_right_open; equality belongs to higher class":
        raise ValueError("数据集边界规则须为：小于分界点归低类，等于分界点归高类")
    boundary = float(edges[0])
    text = f"{boundary:g}"
    return {"dataset_root": str(dataset_root), "dataset_name": dataset_root.name,
            "boundary_mm": boundary, "dataset_tag": f"{text}mm",
            "source_labels": {"0": f"<{text}mm", "1": f">={text}mm"},
            "model_labels": {"0": f">={text}mm", "1": f"<{text}mm"},
            "source_metadata_sha256": sha256(metadata_path),
            "split_indices_sha256": sha256(dataset_root / "split_indices.npz")}


# =========================================================
# 1. 核对固定划分与原始标签（只检查，不重新划分）
# =========================================================
def audit_dataset(dataset_root, require_source_signal=False):
    dataset_root = Path(dataset_root)
    metadata = read_json(dataset_root / "source_metadata.json")
    info = read_dataset_info(dataset_root)
    boundary = info["boundary_mm"]
    manifest = read_json(dataset_root / "provenance/manifest.json")
    records = {record["point"]: record for record in manifest}
    if len(records) != len(manifest):
        raise ValueError("provenance 中存在重复点位")
    seen_ids, seen_indices = set(), set()
    summary = {}
    with np.load(dataset_root / "split_indices.npz", allow_pickle=False) as fixed:
        for split in SPLITS:
            folder = dataset_root / split
            ids = np.load(folder / "arrays/point_ids.npy", allow_pickle=False).tolist()
            indices = np.load(folder / "indices.npy", allow_pickle=False)
            labels = np.load(folder / "y.npy", allow_pickle=False)
            samples = read_json(folder / "samples.json")
            frames = np.load(folder / "arrays/frame_ids.npy", allow_pickle=False)
            if labels.shape != (len(ids),) or not np.isin(labels, [0, 1]).all():
                raise ValueError(f"{split}: 标签形状或数值错误")
            if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                raise ValueError(f"{split}: 点位重复或跨集合泄漏")
            if len(set(indices.tolist())) != len(ids) or seen_indices.intersection(indices):
                raise ValueError(f"{split}: source_index 重复或跨集合泄漏")
            if not np.array_equal(indices, fixed[split]):
                raise ValueError(f"{split}: indices 与固定划分不一致")
            if ids != [r["point_id"] for r in samples]:
                raise ValueError(f"{split}: samples 与数组顺序不一致")
            if not np.array_equal(indices, [r["source_index"] for r in samples]):
                raise ValueError(f"{split}: samples 的 source_index 不一致")
            if not np.array_equal(labels, [r["label"] for r in samples]):
                raise ValueError(f"{split}: samples 的 label 不一致")
            thickness = np.array([r["thickness"] for r in samples])
            if not np.isfinite(thickness).all() or not np.array_equal(labels, thickness >= boundary):
                raise ValueError(f"{split}: 原始标签必须为 0=<{boundary:g}mm，1=≥{boundary:g}mm")
            if len(frames) != 50 or len(set(frames.tolist())) != 50:
                raise ValueError(f"{split}: 应有 50 个唯一帧 ID")
            if not set(ids).issubset(records):
                raise ValueError(f"{split}: 缺少点位追溯记录")
            source_signal = (load_source_signal(folder, len(ids), frames)
                             if require_source_signal else None)
            # 原采样长度来自数据元信息；兼容新600点尾部及历史645点数据。
            for region in ("main", "tail"):
                length = metadata["array_shapes"][f"{region}_signal"][-1]
                signal = np.load(folder / f"arrays/{region}_signal.npy", mmap_mode="r")
                if signal.shape != (len(ids), 50, 2, length) or not np.isfinite(signal).all():
                    raise ValueError(f"{split}/{region}: 信号形状或有限性异常")
                if source_signal is not None:
                    validate_source_region(source_signal, signal, folder, region)
            seen_ids.update(ids)
            seen_indices.update(indices.tolist())
            summary[split] = {"points": len(ids), "source_label_counts":
                              np.bincount(labels, minlength=2).tolist()}
    if len(seen_ids) != metadata["point_count"] or seen_indices != set(range(metadata["point_count"])):
        raise ValueError("固定划分没有恰好覆盖全部源样本")
    return summary


# =========================================================
# 2. 四种输入：mean/std、原始50帧、最大1帧、最大3帧平均
# =========================================================
def validate_frame_array(signal, frame_ids):
    if signal.ndim != 4 or signal.shape[1:3] != (50, 2) or signal.shape[-1] < 1:
        raise ValueError("信号应为 [N,50,2,L]")
    frame_ids = np.asarray(frame_ids)
    if frame_ids.shape != (50,) or len(np.unique(frame_ids)) != 50:
        raise ValueError("frame_ids 应包含 50 个唯一帧 ID")
    if not np.isfinite(signal).all():
        raise ValueError("信号须为有限值")
    return frame_ids


def load_source_signal(folder, point_count, frame_ids):
    """只读完整、分割前的50帧；缺失时不能用主体或尾部代替排名。"""
    path = Path(folder) / "arrays/source_signal.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"方案2/3须在完整原信号上选帧，缺少 {path}。"
            "请使用包含完整原信号的数据集，或在数据集分割工具中补充完整原信号；不能分段排名。")
    signal = np.load(path, mmap_mode="r", allow_pickle=False)
    validate_frame_array(signal, frame_ids)
    if len(signal) != point_count:
        raise ValueError("完整原信号的点位数量与 point_ids 不一致")
    metadata_path = Path(folder) / "metadata.json"
    if metadata_path.is_file():
        original_frames = read_json(metadata_path).get("original_signal", {}).get("frame_ids")
        if original_frames is not None and not np.array_equal(frame_ids, original_frames):
            raise ValueError("完整原信号的帧顺序与分割信号不一致")
    return signal


def validate_source_region(source_signal, signal, folder, region):
    """通过绝对采样索引核对点位、帧和Beam，避免读入错位的分割数组。"""
    indices = np.load(Path(folder) / f"arrays/{region}_absolute_index.npy", allow_pickle=False)
    if (indices.shape != (len(source_signal), 2, signal.shape[-1])
            or indices.dtype.kind not in "iu"
            or np.any(indices < (-1 if region == "tail" else 0))
            or np.any(indices >= source_signal.shape[-1])):
        raise ValueError(f"{region}: 绝对采样索引形状、类型或范围错误")
    valid = indices >= 0
    # -1只表示尾部右侧补零；不能按NumPy负索引误读最后一个真实采样。
    if np.any(np.diff(valid.astype(np.int8), axis=-1) > 0):
        raise ValueError(f"{region}: 只允许在尾部右侧连续补零")
    expected = np.take_along_axis(source_signal, np.maximum(indices, 0)[:, None, :, :], axis=-1)
    expected = np.where(valid[:, None, :, :], expected, 0)
    if not np.array_equal(signal, expected):
        raise ValueError(f"{region}: 分割信号与完整原信号不对应，不能按共用帧索引提取")


def select_frame_indices(source_signal, frame_ids, count):
    """在完整原信号上按 max(abs(x)) 排序，两个Beam及全部采样点共同评分。"""
    frame_ids = validate_frame_array(source_signal, frame_ids)
    if count not in (1, 3):
        raise ValueError("仅支持选最大1帧或最大3帧")
    scores = np.max(np.abs(source_signal.astype(np.float64)), axis=(2, 3))
    # 幅值相同时选真实帧ID较小者，排序不依赖磁盘数组中的帧顺序。
    frame_order = np.argsort(frame_ids, kind="stable")
    order = frame_order[np.argsort(-scores[:, frame_order], axis=1, kind="stable")]
    return order[:, :count]


def build_representation(signal, frame_ids, input_index, seq_len=None, rank_metric=None,
                         *, selected_indices=None):
    """分割信号=[N,50,2,L]；方案2/3只接受完整原信号预先选好的共享帧索引。"""
    cfg = get_cfg(input_index)
    seq_len = cfg.seq_len if seq_len is None else seq_len
    rank_metric = cfg.rank_metric if rank_metric is None else rank_metric
    frame_ids = validate_frame_array(signal, frame_ids)
    if rank_metric != "peak":
        raise ValueError("rank_metric 须为 peak：完整原信号最大绝对幅值选帧")
    if seq_len != FIXED_SEQ_LEN:
        raise ValueError(f"所有输入方案固定 seq_len={FIXED_SEQ_LEN}，不接受其他长度")

    used_ids = np.broadcast_to(frame_ids, (len(signal), 50)).copy()
    if input_index == 0:
        mean = signal.mean(axis=1, dtype=np.float64)
        std = signal.std(axis=1, dtype=np.float64, ddof=0)
        representation = np.concatenate([mean, std], axis=1).astype(np.float32)
    elif input_index == 1:
        representation = signal.astype(np.float32, copy=True)
    else:
        count = 1 if input_index == 2 else 3
        if selected_indices is None:
            raise ValueError("方案2/3必须提供由完整原信号计算的 selected_indices，禁止分段独立选帧")
        selected = np.asarray(selected_indices)
        if (selected.shape != (len(signal), count) or selected.dtype.kind not in "iu"
                or np.any(selected < 0) or np.any(selected >= 50)
                or any(len(set(row.tolist())) != count for row in selected)):
            raise ValueError("共享帧索引的形状、类型、范围或唯一性错误")
        chosen = signal[np.arange(len(signal))[:, None], selected]
        # 保留原波形符号；Top3 对相同三帧的主体/尾部分别逐采样位置平均。
        representation = chosen.mean(axis=1, dtype=np.float64).astype(np.float32)
        used_ids = frame_ids[selected]
    if seq_len != representation.shape[-1]:
        representation = resample(representation, seq_len, axis=-1).astype(np.float32)
    representation = apply_input_window(representation)
    return representation, used_ids


def make_input(dataset_root, split, seq_len=None, input_index=INPUT_INDEX, rank_metric=None):
    if split not in SPLITS:
        raise ValueError("split 无效")
    cfg = get_cfg(input_index)
    dataset_root = Path(dataset_root)
    info = read_dataset_info(dataset_root)
    folder = dataset_root / split
    ids = np.load(folder / "arrays/point_ids.npy", allow_pickle=False).tolist()
    frames = np.load(folder / "arrays/frame_ids.npy", allow_pickle=False)
    data = {"input_index": input_index, "input_name": cfg.input_name,
            "boundary_mm": info["boundary_mm"], "dataset_root": info["dataset_root"],
            "dataset_name": info["dataset_name"],
            "rank_metric": cfg.rank_metric if rank_metric is None else rank_metric}
    selected_indices, source_signal = None, None
    if input_index in (2, 3):
        source_signal = load_source_signal(folder, len(ids), frames)
        selected_indices = select_frame_indices(source_signal, frames, 1 if input_index == 2 else 3)
        data["selected_frame_indices"] = selected_indices
    data["frame_selection_source"] = "arrays/source_signal.npy" if source_signal is not None else "all_frames"
    for region in ("main", "tail"):
        signal = np.load(folder / f"arrays/{region}_signal.npy", allow_pickle=False)
        if len(signal) != len(ids):
            raise ValueError(f"{region}: 信号点位数量不一致")
        if source_signal is not None:
            validate_source_region(source_signal, signal, folder, region)
        data[region], data[f"{region}_frame_ids"] = build_representation(
            signal, frames, input_index, seq_len, rank_metric, selected_indices=selected_indices)
    if not np.array_equal(data["main_frame_ids"], data["tail_frame_ids"]):
        raise ValueError("主体与尾部必须使用完全相同的帧ID")
    # 原始 y 保持不变；仅在进入模型时反转一次，使 1 表示厚度小于当前数据集分界点。
    source_y = np.load(folder / "y.npy", allow_pickle=False)
    samples = read_json(folder / "samples.json")
    thickness = np.array([r["thickness"] for r in samples])
    if (source_y.shape != (len(ids),) or not np.isfinite(thickness).all()
            or ids != [r["point_id"] for r in samples]
            or not np.array_equal(source_y, thickness >= info["boundary_mm"])):
        raise ValueError(f"{split}: 标签/点位顺序与当前厚度分界点不一致")
    data.update({"label": (1 - source_y).astype(np.int64), "point_ids": ids,
                 "indices": np.load(folder / "indices.npy", allow_pickle=False)})
    return data


def input_metadata(data):
    return {"input_index": data["input_index"], "input_name": data["input_name"],
            "window": input_window_metadata(),
            "resampling": input_resampling_metadata(),
            "input_scaling": "none",
            "dataset_info": read_dataset_info(data["dataset_root"]),
            "rank_metric": data["rank_metric"], "point_ids": data["point_ids"],
            "frame_selection_source": data["frame_selection_source"],
            "shared_frame_indices": (data["selected_frame_indices"].tolist()
                                     if "selected_frame_indices" in data else None),
            "shapes": {region: list(data[region].shape) for region in ("main", "tail")},
            "used_frame_ids": {region: data[f"{region}_frame_ids"].tolist()
                               for region in ("main", "tail")}}


if __name__ == "__main__":
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--input-index", type=int, choices=range(4), default=INPUT_INDEX)
    chosen, _ = selector.parse_known_args()
    cfg = get_cfg(chosen.input_index)
    parser = argparse.ArgumentParser("检查最新分割信号，可选导出模型输入", parents=[selector])
    parser.add_argument("--dataset-root", type=Path, default=cfg.dataset_root)
    parser.add_argument("--seq-len", type=int, choices=[FIXED_SEQ_LEN], default=cfg.seq_len)
    parser.add_argument("--rank-metric", choices=("peak",), default=cfg.rank_metric)
    parser.add_argument("--export", type=Path, help="可选：将派生输入另存至此目录")
    args = parser.parse_args()
    summary = audit_dataset(args.dataset_root, require_source_signal=args.input_index in (2, 3))
    info = read_dataset_info(args.dataset_root)
    print(json.dumps({"dataset_info": info, "splits": summary}, ensure_ascii=False, indent=2))
    if args.export:
        args.export.mkdir(parents=True, exist_ok=False)
        for split in SPLITS:
            data = make_input(args.dataset_root, split, args.seq_len,
                              args.input_index, args.rank_metric)
            np.savez_compressed(args.export / f"{split}.npz", **data)
            save_json(args.export / f"{split}_input.json", input_metadata(data))
        save_json(args.export / "metadata.json", {
            "input_index": args.input_index, "representation": cfg.input_name,
            "seq_len": args.seq_len, "rank_metric": args.rank_metric,
            "window": input_window_metadata(),
            "input_scaling": "none",
            "labels": info["model_labels"], "dataset_info": info,
            "dataset": str(args.dataset_root.resolve()), "splits": summary,
        })
