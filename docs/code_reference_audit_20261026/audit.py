"""Read-only functional comparison; writes audit artifacts beside this script.

Run from any directory: python -X utf8 -B <absolute path to audit.py>
No training, source edits, or changes to existing datasets/results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
ROOT = PROJECT.parent
REF = ROOT / "code_reference"
CACHE = REF / "top15_long_training_20261026/data"
OLD_DATA = ROOT / "ViT_reference/dataset"
DATA = PROJECT / "raw_data_relabeled/统一_cls2_thr1.3"
sys.path.insert(0, str(PROJECT))
import emd_pipeline as ep
from vit_pipeline import prepare_vit_sample

# Match the project's documented SciPy-before-Torch initialization order.
ep.resample_signals(np.ones((4, 896), dtype=np.float32), 512)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


snapshot = REF / "methods/six_channel/code_snapshot"
sys.path.insert(0, str(snapshot / "historical"))
import configs
# Same in-memory extension as controlled_driver.bootstrap; no file mutation.
configs.EXPERIMENTS = ("baseline", "energy_similarity", "six_channel", "beam_vote")
rp = module("audit_ref_prepare", snapshot / "experiment_prepare_data.py")
energy = module("audit_ref_energy", REF / "methods/energy_top10/code_snapshot/experiment_prepare_data.py")


def difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    delta = a - b
    aa, bb = a.reshape(len(a), -1), b.reshape(len(b), -1)
    ac, bc = aa - aa.mean(1, keepdims=True), bb - bb.mean(1, keepdims=True)
    denom = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    corr = np.divide((ac * bc).sum(1), denom, out=np.zeros(len(a)), where=denom > 0)
    return {
        "equal": bool(np.array_equal(a, b)), "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
        "sample_corr_median": float(np.median(corr)),
        "sample_corr_p10": float(np.quantile(corr, .1)),
    }


report = {"scope": "current source + unified tasks1-7 versus code_reference frozen four methods",
          "no_training": True, "dataset": {}, "preprocessing": {}, "model": {}}
rows = []
raw_map, ref_map, labels, split_ids = {}, {}, {}, {}
frames = None
for split, old_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
    x, y, records = ep.load_split(DATA, split)
    ids = [r["sample_id"] for r in records]
    split_ids[split] = ids
    old = OLD_DATA / old_split
    old_ids = np.load(old / "arrays/point_ids.npy", allow_pickle=False).tolist()
    original = np.load(old / "arrays/source_signal.npy", mmap_mode="r")
    main = np.load(old / "arrays/main_signal.npy", mmap_mode="r")
    tail = np.load(old / "arrays/tail_signal.npy", mmap_mode="r")
    starts = np.load(old / "arrays/main_absolute_index.npy", mmap_mode="r")[:, :, 0]
    tails = np.load(old / "arrays/tail_absolute_index.npy", mmap_mode="r")
    frames = np.load(old / "arrays/frame_ids.npy", allow_pickle=False)
    old_records = read(old / "samples.json")
    rp.validate_source_region(original, main, old, "main")
    rp.validate_source_region(original, tail, old, "tail")
    raw_stored = np.load(DATA / split / "X.npy", mmap_mode="r")
    report["dataset"][split] = {
        "n": len(ids), "thin_thick_counts": np.bincount(y, minlength=2).tolist(),
        "id_order_equals_old_reference": ids == old_ids,
        "labels_equal_source_labels": bool(np.array_equal(y, np.load(old / "y.npy"))),
        "thickness_equal": all(a["depth_value"] == b["thickness"] for a, b in zip(records, old_records)),
        "adc_cleaned_signal_equal_old_reference": bool(np.array_equal(x, original)),
        "stored_raw_unequal_elements": int(np.count_nonzero(raw_stored != original)),
        "reference_regions_exact_source_slices": True,
    }
    for i, pid in enumerate(ids):
        raw_map[pid] = x[i]
        labels[pid] = int(y[i])
        ref_map[pid] = dict(main=main[i], tail=tail[i], starts=starts[i], tail_indices=tails[i])

all_ids = sum(split_ids.values(), [])
all_x = np.stack([raw_map[p] for p in all_ids])
all_index = {pid: i for i, pid in enumerate(all_ids)}
selected_peak = rp.select_frame_indices(all_x, frames, 1)[:, 0]
selected_energy = energy.select_energy_similarity_frame(all_x, frames)[0][:, 0]
current_inputs, linear_inputs, shared_inputs = [], [], []
cfg = ep.DynamicEnvelopeConfig()
window = ep.tukey_window(512, .3)
for i, pid in enumerate(all_ids):
    x = raw_map[pid]
    current, audit = prepare_vit_sample(x, form="max1", regions=[ep.RegionSpec(0, 5, "whole")],
        channel_mode=3, mapper=ep.DepthMapper(), dynamic_envelope_config=cfg, point_id=pid)
    processed, chosen, span = ep.prepare_frame_input(x, "max1", 3, pid, cfg)
    _, chosen4, span4 = ep.prepare_frame_input(x, "max1", 4, pid, cfg)
    chunks = []
    for region in range(3):
        for ch in range(2):
            if region == 0:
                block = processed[0, ch, span["starts"][ch]:span["ends"][ch]]
            elif region == 1:
                block = processed[0, ch, span["tail_starts"][ch]:span["tail_valid_ends"][ch]]
                block = np.pad(block, (0, int(span["tail_padding"][ch])))
            else:
                block = processed[0, ch]
            chunks.append(rp.resample_time_domain(block) * window)
    shared = []
    for region in ("main", "tail", "whole"):
        block = (x if region == "whole" else ref_map[pid][region])[selected_peak[i]]
        shared.extend(rp.apply_input_window(rp.resample_time_domain(block)))
    current_inputs.append(current[:, 0])
    linear_inputs.append(np.stack(chunks))
    shared_inputs.append(np.stack(shared))
    rows.append(dict(point_id=pid, split=next(s for s in split_ids if pid in split_ids[s]),
        current_frames=chosen[:, 0].tolist(), reference_peak_frame=int(selected_peak[i]),
        reference_energy_frame=int(selected_energy[i]), channel4_frame=int(chosen4[0]),
        channel4_fallback=not span4["channel4_selection"]["used_overlap"],
        current_starts=span["starts"].tolist(), reference_starts=ref_map[pid]["starts"].tolist(),
        current_tail_padding=span["tail_padding"].tolist(),
        reference_tail_padding=np.sum(ref_map[pid]["tail_indices"] < 0, axis=1).tolist(),
        manual_correction=span["manual_correction_applied"].tolist()))
    if (i + 1) % 50 == 0:
        print(f"Preprocessed {i+1}/{len(all_ids)}", flush=True)
current_inputs, linear_inputs, shared_inputs = map(np.stack, (current_inputs, linear_inputs, shared_inputs))

report["preprocessing"]["frames"] = {}
for split, ids in {**split_ids, "all": all_ids}.items():
    subset = [rows[all_index[p]] for p in ids]
    report["preprocessing"]["frames"][split] = dict(n=len(subset),
        different_between_current_beams=sum(r["current_frames"][0] != r["current_frames"][1] for r in subset),
        both_equal_reference_peak=sum(all(f == r["reference_peak_frame"] for f in r["current_frames"]) for r in subset),
        beam1_diff_reference_peak=sum(r["current_frames"][0] != r["reference_peak_frame"] for r in subset),
        beam2_diff_reference_peak=sum(r["current_frames"][1] != r["reference_peak_frame"] for r in subset),
        channel4_diff_reference_energy=sum(r["channel4_frame"] != r["reference_energy_frame"] for r in subset),
        channel4_fallback=sum(r["channel4_fallback"] for r in subset),
        different_boundary_channels=sum(a != b for r in subset for a,b in zip(r["current_starts"], r["reference_starts"])),
        different_tail_padding_channels=sum(a != b for r in subset for a,b in zip(r["current_tail_padding"], r["reference_tail_padding"])))
report["preprocessing"]["input_differences"] = {}
for title, a, b in [("current_vs_reference_peak", current_inputs, shared_inputs),
                     ("cubic_vs_linear_same_current_frames", current_inputs, linear_inputs),
                     ("current_frames_vs_reference_peak_both_linear", linear_inputs, shared_inputs)]:
    report["preprocessing"]["input_differences"][title] = {
        name: difference(a[:, 2*i:2*i+2], b[:, 2*i:2*i+2])
        for i,name in enumerate(["main", "tail", "whole"])}
report["preprocessing"]["tukey_max_error"] = float(np.max(np.abs(window - rp.apply_input_window(np.ones(512, dtype=np.float32)))))

report["cache_reconstruction"] = {}
for method in ["energy_top10", "six_channel", "beam_vote", "six_channel_beam_vote"]:
    method_result = {}
    for split, dirname, npzname, metaname in [
        ("development", CACHE/f"methods/{method}/input", "development.npz", "development.json"),
        ("test", CACHE/f"test_inputs/{method}", "test_arrays.npz", "test_data.json")]:
        meta = read(dirname/metaname)
        ids = meta["point_ids"]
        with np.load(dirname/npzname, allow_pickle=False) as z:
            ix = np.array([all_index[p] for p in ids])
            picks = selected_energy[ix] if method == "energy_top10" else selected_peak[ix]
            detail = dict(n=len(ids), ids_match_set=set(ids)==set(split_ids["test"] if split=="test" else split_ids["train"]+split_ids["val"]),
                labels_semantically_equal=bool(np.array_equal(z["label"], [1-labels[p] for p in ids])),
                selection_indices_exact=bool(np.array_equal(z["selected_frame_indices"][:,0], picks)),
                frame_ids_exact=bool(np.array_equal(z["main_frame_ids"][:,0], frames[picks])), regions={})
            for region in [k for k in ("main", "tail", "whole") if k in z.files]:
                raw = np.stack([(raw_map[p] if region=="whole" else ref_map[p][region])[f]
                                for p,f in zip(ids,picks)])
                reproduced = rp.apply_input_window(rp.resample_time_domain(raw))
                detail["regions"][region] = difference(reproduced, z[region])
            method_result[split] = detail
    report["cache_reconstruction"][method] = method_result

meta_dir = CACHE / "dataset_metadata/手动阈值_1.3_split_60_15_25_seed42_stratified"
report["dataset"]["old_vs_package_metadata_hashes"] = {
    name: sha(OLD_DATA/name)==sha(meta_dir/name) for name in ["source_metadata.json", "split_indices.npz"]}
report["dataset"]["other_dataset_split_comparison"] = {}
for dirname in [PROJECT/"raw_data", PROJECT/"raw_data_relabeled/cls2_thr1.3"]:
    details = {}
    for split in split_ids:
        rec = read(dirname/split/"samples.json")
        ids = [r["sample_id"] for r in rec]
        y = np.load(dirname/split/"y.npy")
        details[split] = dict(n=len(ids), overlap_with_unified_same_split=len(set(ids)&set(split_ids[split])),
            class_counts=np.bincount(y).tolist(), label_disagreements_at_same_point=sum(int(v)!=labels[p] for p,v in zip(ids,y)))
    report["dataset"]["other_dataset_split_comparison"][dirname.name] = details
bone = lambda ids: {p.split('_')[0] for p in ids}
report["dataset"]["shared_bones"] = {"train_test": len(bone(split_ids['train'])&bone(split_ids['test'])),
    "development_test": len(bone(split_ids['train']+split_ids['val'])&bone(split_ids['test'])),
    "test_bones": len(bone(split_ids['test']))}
dev = CACHE/"methods/six_channel/input"
dev_ids = read(dev/"development.json")["point_ids"]
with np.load(dev/"fold_indices.npz") as z:
    report["dataset"]["reference_folds"] = [dict(fold=i,
        train_n=len(z[f'fold{i}_train']), val_n=len(z[f'fold{i}_validation']),
        val_overlap_current_val=len(set(dev_ids[j] for j in z[f'fold{i}_validation']) & set(split_ids['val']))) for i in range(1,6)]

# Same-weight forward comparison distinguishes encoder bugs from protocol differences.
import torch
from vit_model import RegionChannelViT, RegionChannelCrossVoteViT, ViTConfig
torch.set_num_threads(2)
models = module("audit_reference_models", REF/"methods/six_channel_beam_vote/code_snapshot/experiment_models.py")
import scipy
report["runtime"] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__)
for n in [4,6]:
    cfg = ViTConfig(branches=n, streams=1, length=512, num_classes=2,
        patch_size=64, dim=96, depth=4, heads=8, mlp_dim=384, dropout=0.)
    torch.manual_seed(2023)
    current = RegionChannelViT(cfg).eval()
    torch.manual_seed(2023)
    reference = models.ViT(seq_len=512, patch_size=64, dim=96, depth=4, heads=8,
        mlp_dim=384, dropout=0., channels=2, frames=1, split_beams=True, include_whole=n==6).eval()
    translated = {k.replace('encoders.', 'branches.').replace('.position', '.pos'):v for k,v in current.state_dict().items()}
    same_seed_initial_state_equal = all(torch.equal(v, reference.state_dict()[k]) for k,v in translated.items())
    reference.load_state_dict(translated, strict=True)
    x = torch.from_numpy(shared_inputs[:3,:n]).unsqueeze(2)
    with torch.inference_mode():
        ref_kwargs = {'whole_signal':x[:,4:6,0]} if n==6 else {}
        logits = current(x)
        delta = torch.max(torch.abs(logits-reference(x[:,0:2,0],x[:,2:4,0],**ref_kwargs)))
        test_labels = torch.tensor([0, 1, 0])
        label_flip_loss_delta = torch.abs(torch.nn.functional.cross_entropy(logits,test_labels)
            -torch.nn.functional.cross_entropy(logits.flip(1),1-test_labels))
    vote = RegionChannelCrossVoteViT(cfg)
    ref_vote = models.ViT(seq_len=512, patch_size=64, dim=96, depth=4, heads=8,
        mlp_dim=384, dropout=0., channels=2, frames=1, split_beams=True, include_whole=n==6, beam_vote=True)
    count = lambda m: sum(p.numel() for p in m.parameters())
    report["model"][str(n)] = dict(concat_weights_strictly_mappable=True, concat_forward_max_abs=float(delta),
        same_seed_initial_state_equal=same_seed_initial_state_equal,
        concat_parameters=count(current), reference_concat_parameters=count(reference),
        current_vote_parameters=count(vote), reference_vote_parameters=count(ref_vote),
        current_vote_heads=len(vote.heads), reference_vote_heads=len(ref_vote.beam_heads),
        consistently_flipped_label_and_logit_ce_delta=float(label_flip_loss_delta))
report["model"]["loss_example_correct_class_probs_0.9_0.1"] = dict(
    nll_of_mean=float(-np.log(.5)), mean_of_ce=float(-np.mean(np.log([.9,.1]))))

report["current_results"] = read(PROJECT/"experiments/unified_tasks4567_20261026/results_all7.json")
report["checkpoint_selection_examples"] = []
for row in report["current_results"]:
    history = read(Path(row['directory'])/'history.json')
    best_auc = max(history, key=lambda epoch: epoch['val_auc'])
    report["checkpoint_selection_examples"].append(dict(task=row['task'],
        chosen_by_val_loss_epoch=row['best_epoch'], epochs_run=len(history),
        best_observed_val_auc_epoch=best_auc['epoch'],
        best_observed_val_auc=best_auc['val_auc'], accuracy_at_best_auc=best_auc['val_accuracy']))
report["frozen_source_comparison"] = {}
for batch in ['unified_tasks123_20261025', 'unified_tasks4567_20261026']:
    source_dir = PROJECT/'experiments'/batch/'sources'
    report["frozen_source_comparison"][batch] = {
        file.name: sha(file)==sha(PROJECT/file.name)
        for file in source_dir.glob('*.py') if (PROJECT/file.name).exists()}
report["reference_results"] = read(REF/"reference/report/summary.json")
report["source_sha256"] = {str(p.relative_to(ROOT)):sha(p) for p in [
    PROJECT/"emd_pipeline.py", PROJECT/"vit_pipeline.py", PROJECT/"vit_model.py",
    PROJECT/"run_vit_experiments.py", PROJECT/"learning_rate_schedule.py",
    snapshot/"experiment_prepare_data.py", REF/"methods/energy_top10/code_snapshot/experiment_prepare_data.py",
    REF/"methods/six_channel_beam_vote/code_snapshot/experiment_models.py",
    REF/"methods/six_channel_beam_vote/code_snapshot/controlled_driver.py",
    REF/"methods/six_channel_beam_vote/code_snapshot/fit_fold_extensions.py"]}
assert all(report['dataset'][s]['adc_cleaned_signal_equal_old_reference'] for s in split_ids)
assert report['preprocessing']['frames']['all']['different_boundary_channels'] == 0
assert report['preprocessing']['frames']['all']['different_tail_padding_channels'] == 0
for method in report['cache_reconstruction'].values():
    for data in method.values():
        assert data['ids_match_set'] and data['labels_semantically_equal'] and data['selection_indices_exact']
        assert all(region['equal'] for region in data['regions'].values())
assert all(report['model'][str(n)]['concat_forward_max_abs'] == 0 for n in [4, 6])
report['verification_assertions_passed'] = True
for name, value in [("audit_results.json",report), ("sample_comparison.json",rows)]:
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ["current_results","reference_results","source_sha256"]}, ensure_ascii=False, indent=2))
