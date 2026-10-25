"""Copy the existing raw signals into the exact frozen ViT_reference splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from raw_data.relabel_by_thickness import (
    SPLITS, LabelScheme, _check_output_dir, load_pool, write_dataset,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = APP_DIR / "dataset_protocols" / "cls2_thr1.3_unified.json"
DEFAULT_OUTPUT = APP_DIR / "raw_data_relabeled" / "统一_cls2_thr1.3"


def reference_assignment(
    sample_ids: Sequence[str], manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Match IDs, preserving reference row order, and require one exact partition."""
    lookup = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    if len(lookup) != len(sample_ids):
        raise ValueError("source sample IDs are duplicated")
    splits = manifest["splits"]
    if set(splits) != set(SPLITS) or any(not splits[split] for split in SPLITS):
        raise ValueError("reference must contain nonempty train/val/test splits")
    reference_ids = [sample_id for split in SPLITS for sample_id in splits[split]]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference sample IDs are duplicated within or across splits")
    missing = set(reference_ids) - set(lookup)
    extra = set(lookup) - set(reference_ids)
    if missing or extra:
        raise ValueError(f"sample pool mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return {
        split: np.asarray([lookup[sample_id] for sample_id in splits[split]], dtype=np.int64)
        for split in SPLITS
    }


def create_dataset(source_dir: Path, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    # No overwrite option: retain every existing dataset, even on a repeated run.
    output_dir = _check_output_dir(output_dir, source_dir, overwrite=False)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported split manifest schema")
    scheme = LabelScheme(tuple(manifest["thresholds_mm"]))
    if scheme.num_classes != manifest["num_classes"]:
        raise ValueError("thresholds disagree with manifest class count")
    pool = load_pool(source_dir)
    labels = scheme.labels(pool.depth)
    assignment = reference_assignment(pool.sample_ids, manifest)
    for split, rows in assignment.items():
        counts = {str(i): int(np.sum(labels[rows] == i)) for i in range(scheme.num_classes)}
        if counts != manifest["class_distribution"][split]:
            raise ValueError(f"{split}: labels disagree with reference class counts: {counts}")

    provenance = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_name": manifest_path.name,
        "source": manifest["source"],
        "source_files_sha256": manifest["source_files_sha256"],
        "match_key": "sample_id",
        "row_order": "exact reference point_ids.npy order",
        "signal_source": "existing raw_data X.npy, byte values preserved",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete dataset; a failed build removes its own staging files.
    with tempfile.TemporaryDirectory(prefix="_unified_build_", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        result = write_dataset(
            pool, scheme, labels, assignment, staged, source_dir,
            manifest["ratios"], manifest["seed"], "reference_sample_ids",
        )
        metadata = result["metadata"]
        metadata.update({
            "description": "逐 sample_id 沿用 ViT_reference train/validation/test 归属及行顺序；validation 映射为 val。",
            "reference_split": provenance,
            "display_name": manifest["display_name"],
        })
        scheme_data = json.loads((staged / "label_scheme.json").read_text("utf-8"))
        scheme_data.update({"tag": manifest["tag"], "display_name": manifest["display_name"]})
        for name, payload in (("split_metadata.json", metadata), ("label_scheme.json", scheme_data),
                              ("reference_split_manifest.json", manifest)):
            (staged / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
        lines = [
            manifest["display_name"], "",
            "0=薄骨（depth_value < 1.3 mm）；1=厚骨（depth_value >= 1.3 mm）。",
            "按参考样本 ID 逐条匹配，保留参考的 split 和行顺序；不重新抽样。",
            "原始信号逐值复制自本项目 raw_data；ADC 归零仍由读取入口处理。",
            "原数据集保留；独立 tag 为 " + manifest["tag"] + "。",
            "每个 split 包含 X.npy、y.npy、indices.npy、samples.json。",
            "indices.npy 表示本项目原样本池行号，不能直接用作参考包池行号。",
            "参考包训练缓存标签为 0=厚、1=薄，本数据集沿用本项目标签方向。",
            "参考包的 5 折训练不在本次迁移范围内。", "",
        ]
        for split in SPLITS:
            lines.append(f"{split}: {metadata['sample_counts'][split]}, labels={metadata['class_distribution'][split]}")
        lines.extend([
            "", "复现命令（从仓库目录执行，目标必须不存在）：",
            f'python create_unified_dataset.py --source-dir "{source_dir.resolve()}" '
            f'--output-dir "{output_dir}" --split-manifest "{manifest_path.resolve()}"', "",
        ])
        (staged / "README.txt").write_text("\n".join(lines), "utf-8")
        staged.rename(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=APP_DIR / "raw_data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    metadata = create_dataset(args.source_dir, args.output_dir, args.split_manifest)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "sample_counts": metadata["sample_counts"],
                      "class_distribution": metadata["class_distribution"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
