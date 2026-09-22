"""Re-split ``raw_data`` under a new bone-thickness threshold scheme.

``raw_data/{train,val,test}`` only store per-split arrays.  The thickness that
defines the label lives in ``samples.json`` (``depth_value``, the ``总厚度``
column of ``aa_summary.xlsx``), and the three splits are a partition of one
268-sample pool: ``indices.npy`` holds the row of each sample inside that pool
and the three index sets tile ``range(268)``.  This script therefore rebuilds
the pool from the existing splits -- the upstream dataset
(``D:\\BMU\\骨分层\\肩胛骨最终数据集\\...``) no longer exists on disk -- relabels
every sample from the thresholds given on the command line, and writes a fresh
dataset directory that ``emd_pipeline.load_split`` can read unchanged.

Thresholds are **exclusive lower bounds** (``numpy.digitize`` semantics):
with ``--thresholds 1.0`` a sample is class 1 iff ``depth >= 1.0``, which is
exactly the rule the stored ``y.npy`` was built with.  Two thresholds give
three classes, and more thresholds generalise to more classes.

Nothing is written unless ``--output-dir`` is given; ``--dry-run`` (the
default when no output directory is passed) prints the class balance instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.60, 0.15, 0.25)
DEFAULT_SEED = 42
DEFAULT_FEATURE_SHAPE = (50, 2, 896)


# --------------------------------------------------------------------------
# Pool reconstruction
# --------------------------------------------------------------------------


@dataclass
class SamplePool:
    """Every sample in ``raw_data``, in the original pool ordering."""

    frames: np.ndarray  # [N, 50, 2, 896] float32
    depth: np.ndarray  # [N] float64, bone thickness in mm
    sample_ids: list[str]  # [N] unique ids such as ``N20_P2_01``
    records: list[dict[str, Any]]  # [N] original ``samples.json`` entries
    original_split: np.ndarray  # [N] ``<U5`` split name each sample came from
    source_labels: np.ndarray  # [N] int64, labels stored in the old y.npy
    original_rank: np.ndarray  # [N] int64, position inside that split's files

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def _read_split(source_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    split_dir = source_dir / split
    required = ("X.npy", "y.npy", "indices.npy", "samples.json")
    missing = [name for name in required if not (split_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{split_dir} is missing {', '.join(missing)}")

    frames = np.load(split_dir / "X.npy")
    labels = np.load(split_dir / "y.npy").astype(np.int64).ravel()
    indices = np.load(split_dir / "indices.npy").astype(np.int64).ravel()
    records = json.loads((split_dir / "samples.json").read_text(encoding="utf-8"))

    sizes = {frames.shape[0], labels.size, indices.size, len(records)}
    if len(sizes) != 1:
        raise ValueError(
            f"{split}: X.npy/y.npy/indices.npy/samples.json disagree on sample count {sizes}"
        )
    if frames.ndim != 4 or tuple(frames.shape[1:]) != DEFAULT_FEATURE_SHAPE:
        raise ValueError(
            f"{split}: expected X shape [N,{','.join(map(str, DEFAULT_FEATURE_SHAPE))}], "
            f"got {frames.shape}"
        )
    if indices.size and (indices.min() < 0 or indices.max() >= 10_000):
        raise ValueError(f"{split}: indices.npy does not look like pool row numbers")
    return frames, labels, indices, records


def load_pool(source_dir: Path | str) -> SamplePool:
    """Rebuild the full sample pool from the three stored splits.

    Raises on any inconsistency: overlapping pool rows would silently drop a
    sample, and a duplicate ``sample_id`` would make the pool ambiguous.
    """

    source_dir = Path(source_dir)
    blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = []
    seen_rows: dict[int, str] = {}
    for split in SPLITS:
        frames, labels, indices, records = _read_split(source_dir, split)
        for row in indices.tolist():
            if row in seen_rows:
                raise ValueError(
                    f"pool row {row} appears in both {seen_rows[row]} and {split}; "
                    "the splits must tile the pool exactly once"
                )
            seen_rows[row] = split
        blocks.append((split, frames, labels, indices, records))

    total = len(seen_rows)
    expected = set(range(total))
    if set(seen_rows) != expected:
        missing = sorted(expected - set(seen_rows))[:10]
        raise ValueError(f"pool rows are not contiguous 0..{total - 1}; missing e.g. {missing}")

    frames = np.zeros((total, *DEFAULT_FEATURE_SHAPE), dtype=np.float32)
    depth = np.full(total, np.nan, dtype=np.float64)
    sample_ids: list[str | None] = [None] * total
    original_split = np.empty(total, dtype="<U5")
    source_labels = np.full(total, -1, dtype=np.int64)
    original_rank = np.full(total, -1, dtype=np.int64)
    records: list[dict[str, Any] | None] = [None] * total

    for split, block_frames, block_labels, indices, block_records in blocks:
        for position, row in enumerate(indices.tolist()):
            record = block_records[position]
            if "depth_value" not in record:
                raise ValueError(f"{split} row {row}: samples.json entry has no 'depth_value'")
            frames[row] = block_frames[position]
            depth[row] = float(record["depth_value"])
            sample_ids[row] = str(record.get("sample_id", ""))
            records[row] = record
            original_split[row] = split
            source_labels[row] = int(block_labels[position])
            original_rank[row] = position

    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"duplicate sample_id values in the pool: {duplicates[:10]}")
    if any(not sid for sid in sample_ids):
        raise ValueError("some samples.json entries carry an empty sample_id")
    if not np.isfinite(depth).all():
        raise ValueError("some samples have a non-finite depth_value")

    return SamplePool(
        frames=frames,
        depth=depth,
        sample_ids=[str(sid) for sid in sample_ids],
        records=[dict(record) for record in records],
        original_split=original_split,
        source_labels=source_labels,
        original_rank=original_rank,
    )


# --------------------------------------------------------------------------
# Label scheme
# --------------------------------------------------------------------------


@dataclass
class LabelScheme:
    """Thresholds plus the derived class names."""

    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError("at least one threshold is required")
        values = [float(value) for value in self.thresholds]
        if any(not np.isfinite(value) for value in values):
            raise ValueError(f"thresholds must be finite, got {self.thresholds}")
        if any(values[index] >= values[index + 1] for index in range(len(values) - 1)):
            raise ValueError(f"thresholds must be strictly increasing, got {values}")
        self.thresholds = tuple(values)

    @property
    def num_classes(self) -> int:
        return len(self.thresholds) + 1

    @property
    def tag(self) -> str:
        """Short, filesystem-safe identifier such as ``cls2_thr1.3``."""

        formatted = "-".join(_format_number(value) for value in self.thresholds)
        return f"cls{self.num_classes}_thr{formatted}"

    def labels(self, depth: np.ndarray) -> np.ndarray:
        """Class index per sample; class ``k`` means ``thresholds[k-1] <= d < thresholds[k]``."""

        edges = np.asarray(self.thresholds, dtype=np.float64)
        return np.digitize(np.asarray(depth, dtype=np.float64), edges, right=False).astype(np.int64)

    def class_names(self) -> list[str]:
        names: list[str] = []
        for index in range(self.num_classes):
            lower = None if index == 0 else self.thresholds[index - 1]
            upper = None if index == self.num_classes - 1 else self.thresholds[index]
            if lower is None:
                names.append(f"depth < {_format_number(upper)}")
            elif upper is None:
                names.append(f"depth >= {_format_number(lower)}")
            else:
                names.append(f"{_format_number(lower)} <= depth < {_format_number(upper)}")
        return names


def _format_number(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m")


def parse_thresholds(text: str) -> LabelScheme:
    """Parse ``"1.3"`` or ``"1.0,2.0"`` into a :class:`LabelScheme`."""

    pieces = [piece.strip() for piece in str(text).replace(";", ",").split(",")]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        raise argparse.ArgumentTypeError("expected at least one threshold, e.g. 1.3 or 1.0,2.0")
    try:
        return LabelScheme(tuple(float(piece) for piece in pieces))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# --------------------------------------------------------------------------
# Stratified re-split
# --------------------------------------------------------------------------


def allocate_counts(count: int, ratios: Sequence[float]) -> list[int]:
    """Split ``count`` items by ``ratios`` with the largest-remainder rule.

    Flooring each share first and handing the leftovers to the largest
    fractional parts keeps the per-class totals exact and deterministic.
    """

    exact = [count * float(ratio) for ratio in ratios]
    counts = [int(np.floor(value)) for value in exact]
    leftover = count - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        counts[index] += 1
    return counts


def stratified_split(
    labels: np.ndarray,
    ratios: Sequence[float],
    seed: int,
    min_per_split: int = 1,
    allow_small_classes: bool = False,
) -> dict[str, np.ndarray]:
    """Assign pool rows to splits, stratifying on the new labels.

    Each class is shuffled and cut independently, so the class balance of the
    pool is preserved in all three splits.  A class too small to honour
    ``min_per_split`` in the smaller splits is an error unless
    ``allow_small_classes`` is set, because an empty val/test fold silently
    breaks early stopping and the metrics.
    """

    labels = np.asarray(labels, dtype=np.int64).ravel()
    ratios = [float(ratio) for ratio in ratios]
    rng = np.random.default_rng(seed)
    assignment = np.full(labels.size, "", dtype="<U5")
    report: list[dict[str, Any]] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        counts = allocate_counts(int(rows.size), ratios)
        smallest = min(counts[1:])
        if smallest < min_per_split and not allow_small_classes:
            raise ValueError(
                f"class {int(label)} has only {rows.size} samples; the stratified split would "
                f"give {tuple(counts)} (val/test need >= {min_per_split}). Lower the thresholds "
                "range, use fewer classes, or pass --allow-small-classes to write it anyway."
            )
        shuffled = rng.permutation(rows.size)
        cursor = 0
        for split_name, count in zip(SPLITS, counts):
            chunk = rows[shuffled[cursor : cursor + count]]
            assignment[chunk] = split_name
            cursor += count
        report.append({"label": int(label), "pool_count": int(rows.size), "counts": counts})
    if (assignment == "").any():
        raise RuntimeError("internal error: some pool rows were not assigned to a split")
    return {split: np.flatnonzero(assignment == split) for split in SPLITS}


def keep_split_assignment(pool: SamplePool) -> dict[str, np.ndarray]:
    """Reuse the split each sample already lives in, changing labels only.

    Rows are returned in the order they appear in the source files, so
    ``--thresholds 1.0 --keep-split`` reproduces the source byte-for-byte.
    """

    return {
        split: np.flatnonzero(pool.original_split == split)[
            np.argsort(pool.original_rank[pool.original_split == split])
        ]
        for split in SPLITS
    }


# --------------------------------------------------------------------------
# Writing the relabelled dataset
# --------------------------------------------------------------------------


def _check_output_dir(output_dir: Path, source_dir: Path, overwrite: bool) -> Path:
    """Refuse to write anywhere that could clobber the source dataset.

    This is a hard safety rail: ``raw_data`` is not under version control and
    there is no upstream copy left on disk, so an accidental in-place write is
    unrecoverable.  Overlap is rejected even with ``--overwrite``.
    """

    output_dir = output_dir.resolve()
    protected = {source_dir.resolve(), Path(__file__).resolve().parent}
    for guarded in protected:
        if output_dir == guarded or output_dir in guarded.parents or guarded in output_dir.parents:
            raise ValueError(
                f"refusing to write into {output_dir}: it overlaps the source dataset {guarded}. "
                "Pass a directory outside the dataset tree (e.g. raw_data_relabeled/<tag>)."
            )
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    return output_dir


def _summarise_assignment(
    labels: np.ndarray, assignment: dict[str, np.ndarray], ratios: Sequence[float]
) -> dict[str, Any]:
    counts = {split: int(assignment[split].size) for split in SPLITS}
    total = sum(counts.values())
    return {
        "sample_counts": counts,
        "actual_ratios": {split: counts[split] / total for split in SPLITS},
        "requested_ratios": {split: float(ratio) for split, ratio in zip(SPLITS, ratios)},
        "class_distribution": {
            split: {
                str(int(label)): int(np.sum(labels[assignment[split]] == label))
                for label in np.unique(labels)
            }
            for split in SPLITS
        },
    }


def write_dataset(
    pool: SamplePool,
    scheme: LabelScheme,
    labels: np.ndarray,
    assignment: dict[str, np.ndarray],
    output_dir: Path,
    source_dir: Path,
    ratios: Sequence[float],
    seed: int,
    method: str,
) -> dict[str, Any]:
    """Write ``train/val/test`` in the layout ``emd_pipeline.load_split`` reads."""

    report = describe_pool(pool, scheme)
    summary = _summarise_assignment(labels, assignment, ratios)
    rows: list[dict[str, Any]] = []

    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # Written in the order the assignment produced: ``--keep-split`` therefore
        # preserves the source row order exactly (a threshold-only round trip is
        # byte-identical), while ``stratified_split`` already yields ascending rows.
        pool_rows = np.asarray(assignment[split], dtype=np.int64)
        frames = np.ascontiguousarray(pool.frames[pool_rows], dtype=np.float32)
        split_labels = labels[pool_rows].astype(np.int64)
        records: list[dict[str, Any]] = []
        for position, row in enumerate(pool_rows.tolist()):
            record = dict(pool.records[row])
            record["source_label"] = int(pool.source_labels[row])
            record["label"] = int(labels[row])
            records.append(record)
            rows.append(
                {
                    "split": split,
                    "pool_row": row,
                    "sample_id": pool.sample_ids[row],
                    "point_id": record.get("point_id", ""),
                    "depth_value": float(pool.depth[row]),
                    "source_label": int(pool.source_labels[row]),
                    "label": int(labels[row]),
                }
            )
        np.save(split_dir / "X.npy", frames)
        np.save(split_dir / "y.npy", split_labels)
        np.save(split_dir / "indices.npy", pool_rows.astype(np.int64))
        (split_dir / "samples.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    np.savez(
        output_dir / "split_indices.npz",
        **{split: np.asarray(assignment[split], dtype=np.int64) for split in SPLITS},
    )

    metadata = {
        "method": method,
        "group_by_bone": False,
        "description": (
            f"按骨头厚度阈值 {list(scheme.thresholds)} mm 重新定义标签后"
            + ("沿用原 train/val/test 归属" if method == "keep_source_split"
               else "按新标签分层随机划分")
            + "；未按骨头编号分组。"
        ),
        "thresholds_mm": list(scheme.thresholds),
        "threshold_rule": "class = digitize(depth_value, thresholds); thresholds are inclusive lower bounds",
        "num_classes": scheme.num_classes,
        "class_names": scheme.class_names(),
        "random_seed": int(seed),
        "source_dir": str(source_dir.resolve()),
        "source_split_counts": report["source_split_counts"],
        "source_total_samples": report["total_samples"],
        **summary,
        "per_class_pool_counts": {
            entry["name"]: entry["count"] for entry in report["per_class"]
        },
        "feature_shape": list(DEFAULT_FEATURE_SHAPE),
        "x_dtype": "float32",
        "y_dtype": "int64",
        "adc_dc_level_preserved": True,
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "label_scheme.json").write_text(
        json.dumps(
            {
                "thresholds_mm": list(scheme.thresholds),
                "num_classes": scheme.num_classes,
                "tag": scheme.tag,
                "class_names": scheme.class_names(),
                "label_rule": "label = np.digitize(depth_value, thresholds), thresholds inclusive",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_manifest(output_dir / "manifest.csv", rows)
    (output_dir / "README.txt").write_text(
        _readme_text(scheme, report, summary, output_dir, source_dir, seed, ratios, method),
        encoding="utf-8",
    )
    return {"metadata": metadata, "rows": rows}


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    header = ["split", "pool_row", "sample_id", "point_id", "depth_value", "source_label", "label"]
    lines = [",".join(header)]
    for row in sorted(rows, key=lambda item: (SPLITS.index(item["split"]), item["pool_row"])):
        lines.append(
            ",".join(
                [
                    row["split"],
                    str(row["pool_row"]),
                    str(row["sample_id"]),
                    str(row["point_id"]),
                    f"{row['depth_value']:.4f}",
                    str(row["source_label"]),
                    str(row["label"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readme_text(
    scheme: LabelScheme,
    report: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
    source_dir: Path,
    seed: int,
    ratios: Sequence[float],
    method: str,
) -> str:
    thresholds = ", ".join(_format_number(value) for value in scheme.thresholds)
    lines = [
        f"{scheme.num_classes} 分类数据集（骨头厚度阈值 {thresholds} mm）",
        "=" * 40,
        "",
        f"标签规则：depth_value 是骨头厚度（源表 '总厚度' 列，单位 mm，与 896 点采集深度轴无关）。",
        f"阈值取下界（包含）：label = digitize(depth_value, [{thresholds}])。",
        "",
    ]
    for entry in report["per_class"]:
        lines.append(
            f"- label {entry['label']} = {entry['name']}：{entry['count']} 条"
            f"（{entry['share']:.1%}）"
        )
    lines += [
        "",
        f"划分方法：{'沿用原 train/val/test 归属，仅重贴标签' if method == 'keep_source_split' else '按新标签分层随机划分'}",
        "是否按骨头分组：否",
        f"随机种子：{seed}",
        "目标比例：train={:.2f}%, val={:.2f}%, test={:.2f}%".format(*[r * 100 for r in ratios]),
        "",
        "实际样本数：",
    ]
    for split in SPLITS:
        per_class = summary["class_distribution"][split]
        detail = " / ".join(f"label {key}: {value}" for key, value in sorted(per_class.items()))
        lines.append(f"- {split}: {summary['sample_counts'][split]}（{detail}）")
    lines += [
        "",
        "每个子目录均包含 X.npy、y.npy、indices.npy 和 samples.json。",
        "indices.npy 是样本在 268 条完整池中的原始行号；",
        "split_indices.npz 保存三个集合的池行号，split_metadata.json 记录阈值、种子与类别分布，",
        "manifest.csv 给出每个样本的 split / 厚度 / 新旧标签对照。",
        "",
        f"源目录：{source_dir.resolve()}",
        "信号与源数据逐位相同（未做 ADC 直流归零，由 emd_pipeline.load_split 统一处理）。",
        "",
        "复现命令：",
        f"python raw_data/relabel_by_thickness.py --thresholds {thresholds} "
        f"--output-dir {output_dir} --seed {seed} "
        f"--ratios {' '.join(f'{r:g}' for r in ratios)}"
        + ("" if method != "keep_source_split" else " --keep-split"),
        "",
        "训练（示例）：",
        f"python run_emd_experiments.py --data-dir {output_dir} --forms top3_mean \\",
        "    --region-set dyn_envelope --channel-mode 1",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def describe_pool(pool: SamplePool, scheme: LabelScheme) -> dict[str, Any]:
    """Class counts for the whole pool plus the current train/val/test split."""

    labels = scheme.labels(pool.depth)
    names = scheme.class_names()
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        mask = labels == index
        per_class.append(
            {
                "label": index,
                "name": name,
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "depth_min": float(pool.depth[mask].min()) if mask.any() else None,
                "depth_max": float(pool.depth[mask].max()) if mask.any() else None,
                "per_split": {
                    split: int(np.sum(mask & (pool.original_split == split))) for split in SPLITS
                },
            }
        )
    agree = float(np.mean(labels == pool.source_labels))
    return {
        "thresholds": list(scheme.thresholds),
        "num_classes": scheme.num_classes,
        "class_names": names,
        "total_samples": len(pool),
        "depth_range": [float(pool.depth.min()), float(pool.depth.max())],
        "per_class": per_class,
        "source_split_counts": {
            split: int(np.sum(pool.original_split == split)) for split in SPLITS
        },
        "old_label_matches_thresholds": bool(
            np.array_equal(labels, pool.source_labels)
        ),
        "label_agreement_with_source": agree,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"阈值 (exclusive lower bound): {', '.join(_format_number(v) for v in report['thresholds'])}"
        f"  ->  {report['num_classes']} 类",
        f"样本总数: {report['total_samples']}  厚度范围: "
        f"{report['depth_range'][0]:.3f} - {report['depth_range'][1]:.3f} mm",
        "",
        f"{'类':<4}{'区间':<26}{'总数':>6}{'占比':>9}"
        f"{'train':>8}{'val':>7}{'test':>7}{'厚度范围':>18}",
    ]
    for entry in report["per_class"]:
        span = (
            "--"
            if entry["depth_min"] is None
            else f"{entry['depth_min']:.2f}-{entry['depth_max']:.2f}"
        )
        lines.append(
            f"{entry['label']:<4}{entry['name']:<26}{entry['count']:>6}{entry['share']:>8.1%}"
            f"{entry['per_split']['train']:>8}{entry['per_split']['val']:>7}"
            f"{entry['per_split']['test']:>7}{span:>18}"
        )
    smallest = min(entry["count"] for entry in report["per_class"])
    lines.append("")
    lines.append(f"最小类样本数: {smallest}")
    if smallest * 0.25 < 2:
        lines.append("[!] 最小类样本过少，分层三划分会出现空的 val/test 折")
    if report["old_label_matches_thresholds"]:
        lines.append("[ok] 该阈值方案与现存 y.npy 完全一致（可作为回归基线）")
    else:
        agreement = report.get("label_agreement_with_source")
        extra = "" if agreement is None else f"（与现存标签一致率 {agreement:.1%}）"
        lines.append("[!] 该阈值方案会改动现存标签，重新划分后需重跑实验" + extra)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按新的骨头厚度阈值重新划分 raw_data。阈值是下界（包含），"
            "例如 --thresholds 1.0 等价于 label = (depth_value >= 1.0)，"
            "与现存 y.npy 的构造规则相同。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python raw_data/relabel_by_thickness.py --thresholds 1.3 --dry-run\n"
            "  python raw_data/relabel_by_thickness.py --thresholds 1.0,2.0 --dry-run\n"
            "  python raw_data/relabel_by_thickness.py --thresholds 1.3 "
            "--output-dir raw_data_relabeled/1.3\n"
        ),
    )
    parser.add_argument(
        "--thresholds",
        "--threshold",
        dest="thresholds",
        required=True,
        help="逗号分隔的厚度阈值（mm）。1 个=二分类，2 个=三分类，k 个=k+1 类",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="现有 train/val/test 所在目录（默认：本脚本所在目录，即 raw_data）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="新数据集目录；省略或配合 --dry-run 时只打印统计，不写任何文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印统计报告。未给出 --output-dir 时自动开启",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="划分随机种子")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_RATIOS),
        help="train/val/test 目标比例，默认 0.6 0.15 0.25",
    )
    parser.add_argument(
        "--keep-split",
        action="store_true",
        help="不重新划分，沿用现有 train/val/test 归属，只按新阈值重贴标签",
    )
    parser.add_argument(
        "--min-per-split",
        type=int,
        default=1,
        help="分层划分时 val/test 每类的最小样本数，低于该值即报错（默认 1）",
    )
    parser.add_argument(
        "--allow-small-classes",
        action="store_true",
        help="某类样本过少时也照常写出（会产生空的 val/test 折）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录已存在时先删除再写入",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式打印报告（便于脚本消费）",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.scheme = parse_thresholds(args.thresholds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    ratios = [float(value) for value in args.ratios]
    if any(value <= 0 for value in ratios):
        parser.error("--ratios must all be positive")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        parser.error(f"--ratios must sum to 1.0, got {total}")
    args.ratios = ratios
    if args.dry_run:
        args.output_dir = None
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is not None:
        # Fail before doing any work, and never return a directory we would have
        # to delete: the guard rejects overlaps even under --overwrite.
        args.output_dir = _check_output_dir(args.output_dir, args.source_dir, args.overwrite)
    pool = load_pool(args.source_dir)
    scheme: LabelScheme = args.scheme
    labels = scheme.labels(pool.depth)
    report = describe_pool(pool, scheme)

    if args.output_dir is None:
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_report(report))
            print("\n（未给出 --output-dir：仅预览统计，未写出任何文件）")
        return 0

    assignment = (
        keep_split_assignment(pool)
        if args.keep_split
        else stratified_split(
            labels,
            args.ratios,
            args.seed,
            min_per_split=args.min_per_split,
            allow_small_classes=args.allow_small_classes,
        )
    )
    method = "keep_source_split" if args.keep_split else "stratified_random_split"
    result = write_dataset(
        pool,
        scheme,
        labels,
        assignment,
        args.output_dir.resolve(),
        args.source_dir,
        args.ratios,
        args.seed,
        method,
    )
    print(format_report(report))
    summary = result["metadata"]
    print("")
    print(f"已写出：{args.output_dir.resolve()}")
    for split in SPLITS:
        print(
            f"  {split}: {summary['sample_counts'][split]}  "
            f"{summary['class_distribution'][split]}"
        )
    print("  （阈值规则：label = digitize(depth_value, thresholds)，阈值为下界）")
    return 0


if __name__ == "__main__":
    # The Windows console defaults to a legacy code page; degrade instead of
    # crashing on any character it cannot encode (the dataset paths are Chinese).
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
