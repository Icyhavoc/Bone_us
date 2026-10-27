"""Explicitly requested test-accuracy comparison of frozen screening models; no training."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import concurrent.futures as cf
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

METHODS = ('energy_top10', 'six_channel', 'beam_vote', 'six_channel_beam_vote')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_parent(source, expected=None):
    plan = read(source / 'run_plan.json')
    assert read(source / 'status.json')['phase'] == 'screen_complete'
    assert read(source / 'result_audit.json')['passed']
    assert sha(source / 'run_plan.json') == read(source / 'status.json')['run_plan_sha256']
    for relative, digest in plan['frozen_files'].items():
        assert sha(source / relative) == digest, relative
    for relative, digest in plan['source_data_sha256'].items():
        assert sha(Path(plan['dataset_root']) / relative) == digest, relative
    if expected:
        for relative, digest in expected.items():
            assert sha(source / relative) == digest, relative
    return plan


def prepare(source, output):
    source = source.resolve()
    plan = verify_parent(source)
    baseline = Path(plan['source_run']) / 'all_config_tests/20261022_test_accuracy'
    assert read(baseline / 'verification.json')['status'] == 'passed'
    cases = {}
    for method in METHODS:
        rows = []
        for candidate in plan['candidate_ids']:
            entry = source / 'methods' / method / 'screen' / candidate
            result = read(entry / 'result.json')
            origin = Path(read(entry / 'reused_from.json')['directory']) if result['reused'] else entry
            frozen = read(origin / 'frozen_selection.json')
            assert sha(origin / 'config.json') == frozen['config_sha256']
            reference = origin if result['reused'] else None
            rows.append(dict(id=candidate, parameters=result['parameters'], directory=str(origin),
                checkpoints={str(origin / key): digest for key, digest in frozen['checkpoints'].items()},
                config_sha256=frozen['config_sha256'], selection_sha256=sha(origin / 'frozen_selection.json'),
                reference_test_directory=str(reference) if reference else None,
                reference_files_sha256={p.name: sha(p) for p in reference.glob('test_*') if p.is_file()} if reference else {}))
        cases[method] = rows
    original = read(baseline / 'manifest.json')
    cases['baseline'] = []
    for case in original['cases']:
        if case['stage'] != 'screen':
            continue
        origin = Path(plan['source_run']) / 'screen' / case['id']
        reference = baseline / 'screen' / case['id']
        cases['baseline'].append(dict(id=case['id'], parameters=case['parameters'], directory=str(origin),
            checkpoints={str(Path(plan['source_run']) / key): value for key, value in case['checkpoints'].items()},
            config_sha256=sha(origin / 'config.json'), reference_test_directory=str(reference),
            reference_files_sha256={p.name: sha(p) for p in reference.iterdir() if p.is_file()}))
    assert all(len(rows) == 256 for rows in cases.values())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'evaluator.py')
    manifest = dict(created_at=now(), source_run=str(source), baseline_reference=str(baseline),
        parent_files_sha256={name: sha(source / name) for name in ('run_plan.json', 'result_audit.json', 'review_candidates.json', 'launcher.py')},
        baseline_cache_sha256={name: sha(baseline / name) for name in ('test_arrays.npz', 'test_data.json', 'manifest.json', 'verification.json')},
        evaluator_sha256=sha(output / 'evaluator.py'), cases=cases, test_points=67, threshold=0.5,
        precision='bf16', inference_batch_size=32, ensemble='arithmetic mean of five fold probabilities',
        beam_ensemble='equal mean of two Beam probabilities', training_performed=False,
        new_variant_config_evaluations=964, reused_variant_evaluations=60, reused_baseline_evaluations=256,
        preflight_repeated_fold_inferences=20, test_used_for_ranking=True, test_used_for_epoch_or_threshold_selection=False,
        ranking_rule='correct count descending; equal counts share rank; ID only determines display order',
        user_request='不看auc，应该看在测试集上的正确率', interpretation='Exploratory comparison on a previously inspected test set')
    save(output / 'manifest.json', manifest)
    save(output / 'status.json', dict(phase='prepared', completed=0, total=1280, created_at=now()))
    save(source / 'latest_test_accuracy_evaluation.json', dict(directory=str(output), status_file=str(output / 'status.json')))
    return output


def candidate_checks(case):
    directory = Path(case['directory'])
    assert sha(directory / 'config.json') == case['config_sha256']
    if 'selection_sha256' in case:
        assert sha(directory / 'frozen_selection.json') == case['selection_sha256']
    for path, digest in case['checkpoints'].items():
        assert sha(path) == digest, path
    if case['reference_test_directory']:
        for name, digest in case['reference_files_sha256'].items():
            assert sha(Path(case['reference_test_directory']) / name) == digest, name


def infer(case, method, history, training, data, progress):
    import numpy as np
    import torch
    candidate_checks(case)
    cfg = read(Path(case['directory']) / 'config.json')
    args = argparse.Namespace(**cfg)
    regions = ('main', 'tail', 'whole') if method in ('six_channel', 'six_channel_beam_vote') else ('main', 'tail')
    inputs = [torch.as_tensor(data[key], device='cuda') for key in regions]
    vote = method in ('beam_vote', 'six_channel_beam_vote')
    folds, beam_folds = [], []
    for fold, path in enumerate(case['checkpoints'], 1):
        progress(current_candidate=case['id'], current_fold=fold, last_actual_progress=now())
        model, optimizer = history.setup_model(args)
        del optimizer  # No optimizer state or training step is used in this evaluation.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint.get('scaling') is None
        for key in ('seq_len', 'patch_size', 'dim', 'depth', 'heads', 'mlp_dim', 'split_beams'):
            assert checkpoint['config'][key] == cfg[key]
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        probabilities, beams = [], []
        for start in range(0, len(data['label']), 32):
            tensors = [value[start:start + 32] for value in inputs]
            p = history.gpu_predict(model, *tensors)
            probabilities.append(p)
            if vote:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = (model(tensors[0], tensors[1], whole_signal=tensors[2], return_beam_logits=True)
                        if len(tensors) == 3 else model(*tensors, return_beam_logits=True)).float()
                branch_p = logits.softmax(-1)[:, :, 1].cpu().numpy()
                np.testing.assert_array_equal(p, branch_p.mean(1))
                beams.append(branch_p)
        folds.append(np.concatenate(probabilities))
        if beams:
            beam_folds.append(np.concatenate(beams))
        assert all(parameter.grad is None for parameter in model.parameters())
        del model
        torch.cuda.empty_cache()
    return np.asarray(folds), np.asarray(beam_folds) if vote else None


def export_case(output, method, case, data, by_fold, beams, training, reused):
    import numpy as np
    assert by_fold.shape == (5, 67) and np.isfinite(by_fold).all()
    assert ((0 <= by_fold) & (by_fold <= 1)).all()
    dest = output / 'results' / method / case['id']
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / 'test_probabilities_by_fold.npy', by_fold)
    if beams is not None:
        assert beams.shape == (5, 67, 2)
        np.testing.assert_array_equal(by_fold, beams.mean(2))
        np.save(dest / 'test_beam_probabilities_by_fold.npy', beams)
    mean = by_fold.mean(0)
    metrics = training.compute_metrics(data['label'], mean, .5)
    training.export_predictions(dest / 'test_predictions.csv', data, mean, .5)
    with (dest / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['point_id'] for row in rows] == data['point_ids']
    np.testing.assert_array_equal(np.array([float(row['probability_below_boundary']) for row in rows], dtype=np.float32), mean)
    assert int(((mean >= .5).astype(int) == data['label']).sum()) == metrics['correct']
    save(dest / 'result.json', dict(method=method, id=case['id'], parameters=case['parameters'], metrics=metrics,
        threshold=.5, ensemble_folds=5, training_performed=False, reused_test_predictions=reused,
        checkpoint_hashes_verified=True, csv_metrics_recomputed=True, test_used_for_ranking=True,
        test_used_for_epoch_or_threshold_selection=False, checkpoint_hashes=case['checkpoints'],
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir() if p.suffix in ('.npy', '.csv')}, completed_at=now()))


def worker(output, method):
    import numpy as np
    manifest = read(output / 'manifest.json')
    source = Path(manifest['source_run'])
    launcher = load('frozen_variant_launcher', source / 'launcher.py')
    bootstrap_method = 'energy_top10' if method == 'baseline' else method
    driver, history, prep, training, cv, extension, args = launcher.initialize(source, bootstrap_method, 'random_0113')
    baseline = Path(manifest['baseline_reference'])
    for name, digest in manifest['baseline_cache_sha256'].items():
        assert sha(baseline / name) == digest
    base_data = read(baseline / 'test_data.json')
    with np.load(baseline / 'test_arrays.npz', allow_pickle=False) as arrays:
        base_data.update({key: arrays[key] for key in arrays.files})
    if method == 'baseline':
        data = base_data
    else:
        data = driver.inputs(prep, extension, Path(args.dataset_root), 'test', launcher.METHODS[method])
    fold_plan = read(source / 'methods' / bootstrap_method / 'input/fold_plan.json')
    assert data['point_ids'] == fold_plan['heldout_test_point_ids'] == base_data['point_ids']
    assert not set(data['point_ids']).intersection(fold_plan['development_point_ids'])
    np.testing.assert_array_equal(data['label'], base_data['label'])
    assert len(data['label']) == 67 and np.bincount(data['label']).tolist() == [26, 41]
    if method not in ('baseline', 'energy_top10'):
        for key in ('main', 'tail', 'selected_frame_indices'):
            np.testing.assert_array_equal(data[key], base_data[key])
    dest = output / 'inputs' / method
    dest.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    np.savez(dest / 'test_arrays.npz', **arrays)
    save(dest / 'test_data.json', {key: value for key, value in data.items() if key not in arrays})
    state = dict(method=method, phase='preflight', completed=0, total=256, started_at=now())
    def progress(**fields):
        state.update(fields, updated_at=now())
        save(output / 'workers' / f'{method}.json', state)
    if method != 'baseline':
        reference = next(case for case in manifest['cases'][method] if case['id'] == 'random_0113')
        probabilities, beams = infer(reference, method, history, training, data, progress)
        origin = Path(reference['reference_test_directory'])
        np.testing.assert_array_equal(probabilities, np.load(origin / 'test_probabilities_by_fold.npy'))
        if beams is not None:
            np.testing.assert_array_equal(beams, np.load(origin / 'test_beam_probabilities_by_fold.npy'))
        save(output / 'preflight' / f'{method}.json', dict(passed=True, candidate='random_0113',
            five_fold_test_probabilities_exact=True, equal_beam_probability_average_verified=beams is not None,
            heldout_point_order_verified=True, inference_batch_size=32, threshold=.5, completed_at=now()))
    progress(phase='test_inference')
    for index, case in enumerate(manifest['cases'][method], 1):
        if not (output / 'results' / method / case['id'] / 'result.json').exists():
            reused = case['reference_test_directory'] is not None
            if reused:
                candidate_checks(case)
                origin = Path(case['reference_test_directory'])
                by_fold = np.load(origin / 'test_probabilities_by_fold.npy')
                beams_path = origin / 'test_beam_probabilities_by_fold.npy'
                beams = np.load(beams_path) if beams_path.exists() else None
                prior = read(origin / ('result.json' if method == 'baseline' else 'test_result.json'))
                assert training.compute_metrics(data['label'], by_fold.mean(0), .5) == prior['metrics']
                with (origin / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                    assert [row['point_id'] for row in csv.DictReader(stream)] == data['point_ids']
            else:
                by_fold, beams = infer(case, method, history, training, data, progress)
            export_case(output, method, case, data, by_fold, beams, training, reused)
        progress(completed=index, current_candidate=case['id'], current_fold=5, last_actual_progress=now())
        if index % 16 == 0:
            print(f'[TEST] {method} {index}/256', flush=True)
    progress(phase='complete')


def final_audit(output):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix
    manifest = read(output / 'manifest.json')
    source = Path(manifest['source_run'])
    verify_parent(source, manifest['parent_files_sha256'])
    assert sha(output / 'evaluator.py') == manifest['evaluator_sha256']
    total, reused = 0, 0
    for method in ('baseline',) + METHODS:
        with np.load(output / 'inputs' / method / 'test_arrays.npz') as arrays:
            y = arrays['label']
        rows = []
        for case in manifest['cases'][method]:
            dest = output / 'results' / method / case['id']
            result = read(dest / 'result.json')
            for name, digest in result['artifacts_sha256'].items():
                assert sha(dest / name) == digest
            by_fold = np.load(dest / 'test_probabilities_by_fold.npy')
            p = by_fold.mean(0); pred = (p >= .5).astype(int)
            assert accuracy_score(y, pred) == result['metrics']['accuracy']
            assert confusion_matrix(y, pred, labels=[0, 1]).tolist() == result['metrics']['confusion_matrix']
            assert int((y == pred).sum()) == result['metrics']['correct']
            if method in ('beam_vote', 'six_channel_beam_vote'):
                np.testing.assert_array_equal(by_fold, np.load(dest / 'test_beam_probabilities_by_fold.npy').mean(2))
            rows.append(result); total += 1; reused += int(result['reused_test_predictions'])
        rows.sort(key=lambda row: (-row['metrics']['correct'], row['id']))
        previous = None
        for position, row in enumerate(rows, 1):
            if previous != row['metrics']['correct']:
                rank = position; previous = row['metrics']['correct']
            row.update(accuracy_rank=rank, display_position=position)
        save(output / 'rankings' / f'{method}.json', rows)
    assert total == 1280 and reused == 316
    assert all(read(output / 'preflight' / f'{method}.json')['passed'] for method in METHODS)
    save(output / 'result_audit.json', dict(passed=True, configuration_results=1280, newly_evaluated_variant_configs=964,
        reused_variant_results=60, reused_baseline_results=256, fold_model_references_verified=6400,
        new_fold_model_inferences=4820, preflight_repeat_fold_inferences=20, preflight_exact_reproductions=4,
        training_performed=False, threshold=.5, all_accuracy_confusion_matrices_recomputed=True,
        two_beam_equal_average_verified=True, input_code_and_original_review_selection_unchanged=True,
        accuracy_only_ranking=True, completed_at=now()))


def execute(output):
    import msvcrt
    lock = (output / 'runner.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        manifest = read(output / 'manifest.json')
        assert sha(output / 'evaluator.py') == manifest['evaluator_sha256']
        def dispatch(method):
            log = output / 'logs' / f'{method}.log'; log.parent.mkdir(exist_ok=True)
            with log.open('a', encoding='utf-8') as stream:
                process = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-u', str(output / 'evaluator.py'),
                    '--worker', method, '--output', str(output)], stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                return method, process.wait()
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(dispatch, method): method for method in METHODS + ('baseline',)}
            while pending:
                done, _ = cf.wait(pending, timeout=10, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    method, code = future.result(); del pending[future]
                    if code:
                        raise RuntimeError(f'{method} exited {code}; inspect logs/{method}.log')
                    print('[METHOD COMPLETE]', method, flush=True)
                completed = len(list((output / 'results').glob('*/*/result.json')))
                workers = {p.stem: read(p) for p in (output / 'workers').glob('*.json')}
                save(output / 'status.json', dict(phase='test_inference', completed=completed, total=1280,
                    workers=workers, updated_at=now(), training_performed=False))
        save(output / 'status.json', dict(phase='audit', completed=1280, total=1280, updated_at=now()))
        final_audit(output)
        save(output / 'status.json', dict(phase='complete', completed=1280, total=1280, updated_at=now(), training_performed=False))
        print('[COMPLETE] Test accuracy audit passed; 964 new evaluations and 316 reused results; no training', flush=True)
    except BaseException:
        save(output / 'error.json', dict(traceback=traceback.format_exc(), created_at=now()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=METHODS + ('baseline',))
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.worker:
        return worker(output, args.worker)
    if not args.prepared:
        output = prepare(args.source, output)
        print('[PREPARED]', output, flush=True)
    if not args.prepare_only:
        execute(output)


if __name__ == '__main__':
    main()
