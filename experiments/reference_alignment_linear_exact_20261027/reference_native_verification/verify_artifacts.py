from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
LINEAR = HERE.parent
EXPERIMENTS = LINEAR.parent
CUBIC = EXPERIMENTS / "reference_alignment_full_20261027"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(directory: Path) -> dict[str, object]:
    manifest = read_json(directory / "manifest.json")
    result = read_json(directory / "result.json")
    for name, expected in manifest["source_sha256"].items():
        actual = sha256(directory / "sources" / name)
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {directory.name}/{name}")
    for item in result["folds"]:
        fold = int(item["fold"])
        checkpoint = directory / f"fold_{fold}" / "best.pt"
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {directory.name}/fold_{fold}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        if not state or not all(torch.isfinite(value).all() for value in state.values()):
            raise AssertionError(f"invalid checkpoint tensors: {directory.name}/fold_{fold}")
    return {
        "directory": directory.name,
        "correct": result["test_metrics_reference_labels"]["correct"],
        "auc": result["test_metrics_reference_labels"]["auc"],
        "folds": len(result["folds"]),
    }


def main() -> None:
    summary = [verify_run(CUBIC), verify_run(LINEAR)]

    project_by_fold = np.load(LINEAR / "test_probabilities_by_fold.npy")[:, :, 1]
    native_by_fold = np.load(HERE / "test" / "test_probabilities_by_fold.npy")
    np.testing.assert_array_equal(project_by_fold, native_by_fold)

    project_beams = np.load(LINEAR / "test_beam_probabilities_by_fold.npy")[:, :, :, 1]
    native_beams = np.load(HERE / "test" / "test_beam_probabilities_by_fold.npy")
    np.testing.assert_array_equal(project_beams, native_beams)

    for fold in range(1, 6):
        project = np.load(LINEAR / f"fold_{fold}" / "validation_probabilities.npy")[:, 1]
        native = np.load(HERE / "folds" / f"fold_{fold}" / "validation_probabilities.npy")
        np.testing.assert_array_equal(project, native)

    native_result = read_json(HERE / "test" / "result.json")
    for name, expected in native_result["artifacts_sha256"].items():
        if sha256(HERE / "test" / name) != expected:
            raise AssertionError(f"native test artifact hash mismatch: {name}")

    print(json.dumps({
        "passed": True,
        "runs": summary,
        "reference_native_probabilities_exact": True,
        "reference_native_correct": native_result["metrics"]["correct"],
        "reference_native_auc": native_result["metrics"]["roc_auc"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
