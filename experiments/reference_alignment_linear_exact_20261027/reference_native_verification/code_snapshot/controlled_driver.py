"""Reproduce random_0130 and compare four time-domain experiments with its frozen trainer."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import ast
import csv
import difflib
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / '超参数探索/max_cv_runs/max4vit_256_cv5_20260928_185341'
EXPERIMENTS = ('baseline', 'energy_similarity', 'six_channel', 'beam_vote', 'six_channel_beam_vote')
JOBS = ('fft_reference',) + EXPERIMENTS
PARAMETERS = ('seq_len', 'patch_size', 'dim', 'depth', 'heads', 'mlp_dim', 'dropout',
              'lr', 'weight_decay', 'batch_size', 'label_smoothing', 'warmup',
              'epochs', 'patience', 'seeds', 'channels', 'frames', 'split_beams')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def status(run, **fields):
    path = run / 'status.json'
    value = read(path) if path.exists() else {}
    value.update(fields, updated_at_utc=now())
    save(path, value)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(run):
    snapshot = run / 'code_snapshot'
    legacy = snapshot / 'historical'
    sys.path.insert(0, str(legacy))
    import configs
    configs.EXPERIMENTS = EXPERIMENTS  # Only needed by the separate experiment preprocessing module.
    import prepare_data as preparation
    import train_minimal as training
    import train_crossval as cv
    history = load_module('historical_search', legacy / '超参数探索/max_cv_search.py')
    extension = load_module('experiment_preparation', snapshot / 'experiment_prepare_data.py')
    models = load_module('experiment_models', snapshot / 'experiment_models.py')
    # Extra point arrays are sliced only when present. Baseline takes the exact historical path.
    cv.POINT_ARRAYS += ('whole', 'whole_frame_ids', 'energy_similarity_scores')
    return history, preparation, training, cv, extension, models


def inputs(preparation, extension, dataset, split, experiment):
    if experiment == 'six_channel_beam_vote':
        experiment = 'six_channel'  # Identical six-channel preprocessing.
    if experiment in ('energy_similarity', 'six_channel'):
        return extension.make_input(dataset, split, input_index=2, experiment=experiment)
    if experiment == 'fft_reference':
        return preparation.make_input(dataset, split, input_index=2)
    original = preparation.resample
    def time_resample(signal, num, axis=-1):
        if axis != -1:
            raise ValueError('Only the historical final sample axis is supported')
        return extension.resample_time_domain(signal, num)
    preparation.resample = time_resample
    try:
        return preparation.make_input(dataset, split, input_index=2)
    finally:
        preparation.resample = original


def resampling(experiment):
    return {'method': 'scipy_fft' if experiment == 'fft_reference' else 'time_domain_linear',
            'target_length': 512, 'axis': -1}


def adapt_variants(history, training, models, experiment, run):
    """Baseline and energy selection never replace the historical fit/model/predict functions."""
    if experiment not in ('six_channel', 'beam_vote', 'six_channel_beam_vote'):
        return
    import torch
    def fetch(args):
        return models.ViT(seq_len=args.seq_len, patch_size=args.patch_size,
            num_classes=args.num_classes, dim=args.dim, depth=args.depth, heads=args.heads,
            mlp_dim=args.mlp_dim, channels=args.channels, dropout=args.dropout,
            frames=args.frames, main_seq_len=args.main_seq_len, tail_seq_len=args.tail_seq_len,
            split_beams=args.split_beams, include_whole=experiment in ('six_channel', 'six_channel_beam_vote'),
            beam_vote=experiment in ('beam_vote', 'six_channel_beam_vote'))
    training.fetch_model = fetch
    def active_regions(args):
        return ('main', 'tail', 'whole') if experiment in ('six_channel', 'six_channel_beam_vote') else ('main', 'tail')
    def controlled_logits(model, tensors, rows, args):
        if experiment in ('six_channel', 'six_channel_beam_vote'):
            return model(tensors[0][rows], tensors[1][rows], whole_signal=tensors[2][rows], return_beam_logits=experiment == 'six_channel_beam_vote')
        return model(tensors[0][rows], tensors[1][rows], return_beam_logits=True)
    class BeamMeanCE(torch.nn.Module):
        def forward(self, logits, labels):
            return (torch.nn.functional.cross_entropy(logits[:, 0], labels)
                    + torch.nn.functional.cross_entropy(logits[:, 1], labels)) / 2
    def controlled_criterion(args):
        return BeamMeanCE() if experiment in ('beam_vote', 'six_channel_beam_vote') else torch.nn.CrossEntropyLoss(label_smoothing=0.)
    def variant_predict(model, *tensors):
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if experiment in ('six_channel', 'six_channel_beam_vote'):
                logits = model(tensors[0], tensors[1], whole_signal=tensors[2], return_beam_logits=experiment == 'six_channel_beam_vote').float()
            else:
                logits = (model(tensors[0], tensors[1], whole_signal=tensors[2], return_beam_logits=True).float() if experiment == 'six_channel_beam_vote' else model(*tensors, return_beam_logits=True).float())
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite logits')
        if experiment in ('beam_vote', 'six_channel_beam_vote'):
            return logits.softmax(-1)[:, :, 1].mean(1).cpu().numpy()
        return logits.softmax(1)[:, 1].cpu().numpy()
    history.__dict__.update(active_regions=active_regions, controlled_logits=controlled_logits,
        controlled_criterion=controlled_criterion, gpu_predict=variant_predict)
    source = (run / 'code_snapshot/fit_fold_extensions.py').read_text(encoding='utf-8')
    exec(compile(source, str(run / 'code_snapshot/fit_fold_extensions.py'), 'exec'), history.__dict__)


def prepare_worker(run):
    import numpy as np
    history, prep, training, cv, extension, models = bootstrap(run)
    plan = read(run / 'run_plan.json')
    historical = Path(plan['historical_run'])
    dataset = Path(plan['dataset_root'])
    original_fit = inspect.getsource(history.fit_fold)
    extended_fit = original_fit
    substitutions = [
        ("for k in ('main','tail')", 'for k in active_regions(args)', 2),
        ('criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.)', 'criterion=controlled_criterion(args)', 1),
        ('model(x[0][rows],x[1][rows])', 'controlled_logits(model,x,rows,args)', 1),
    ]
    for before, after, expected in substitutions:
        assert extended_fit.count(before) == expected, before
        extended_fit = extended_fit.replace(before, after)
    (run / 'code_snapshot/fit_fold_extensions.py').write_text(extended_fit, encoding='utf-8')
    (run / 'fit_fold_extensions.diff').write_text(''.join(difflib.unified_diff(
        original_fit.splitlines(True), extended_fit.splitlines(True),
        fromfile='historical fit_fold (baseline/energy)', tofile='six_channel/beam_vote only')), encoding='utf-8')
    old_tree = ast.parse((run / 'code_snapshot/historical/models/vit_1d_cls.py').read_text(encoding='utf-8-sig'))
    new_tree = ast.parse((run / 'code_snapshot/experiment_models.py').read_text(encoding='utf-8-sig'))
    branch = lambda tree: ast.dump(next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Branch'))
    assert branch(old_tree) == branch(new_tree), 'Transformer encoder changed'
    records = []
    for experiment in JOBS:
        directory = run / experiment
        directory.mkdir()
        train = inputs(prep, extension, dataset, 'train', experiment)
        validation = inputs(prep, extension, dataset, 'validation', experiment)
        data = cv.merge_development(train, validation)
        if experiment == 'baseline':
            for split, expected in [('train', train), ('validation', validation)]:
                actual = extension.make_input(dataset, split, input_index=2)
                for key in ('main', 'tail', 'selected_frame_indices', 'label', 'indices'):
                    np.testing.assert_array_equal(actual[key], expected[key])
        if experiment == 'fft_reference':
            with np.load(historical / 'development.npz', allow_pickle=False) as reference:
                for key in reference.files:
                    np.testing.assert_array_equal(data[key], reference[key])
            assert data['point_ids'] == read(historical / 'development.json')['point_ids']
        if experiment in ('six_channel', 'beam_vote'):
            with np.load(run / 'baseline/development.npz', allow_pickle=False) as baseline:
                for key in ('main', 'tail', 'selected_frame_indices', 'label', 'indices'):
                    np.testing.assert_array_equal(data[key], baseline[key])
        assert data['point_ids'] == read(historical / 'fold_plan.json')['development_point_ids']
        arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
        np.savez(directory / 'development.npz', **arrays)
        save(directory / 'development.json', {key: value for key, value in data.items() if key not in arrays})
        for filename in ('fold_indices.npz', 'fold_plan.json'):
            shutil.copy2(historical / filename, directory / filename)
        save(directory / 'protocol.json', {'dataset_root': str(dataset)})
        metadata = extension.input_metadata(data)
        metadata['resampling'] = resampling(experiment)
        save(directory / 'development_input.json', {'input': metadata,
            'actual_resampling': resampling(experiment), 'experiment': experiment})
        records.append({'experiment': experiment, 'points': len(data['label']),
            'regions': {key: list(data[key].shape) for key in ('main', 'tail', 'whole') if key in data}})
    save(run / 'preflight.json', {'passed': True, 'historical_fft_cache_exact': True,
        'baseline_preprocessing_only_replaces_resample': True, 'encoders_ast_identical': True,
        'all_fold_plans_copied_verbatim': True, 'inputs': records,
        'baseline_and_energy_fit_fold': 'unmodified historical function',
        'six_and_vote_fit_fold': 'only input routing, vote loss and prediction extensions; diff saved'})
    plan['frozen_files'] = {str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()
        and (p.is_relative_to(run / 'code_snapshot') or p.name in ('development.npz', 'development.json',
             'fold_indices.npz', 'fold_plan.json', 'protocol.json'))}
    save(run / 'run_plan.json', plan)
    print('[PREFLIGHT] historical source/cache and time-domain input equivalence verified', flush=True)


def initialize_worker(run, experiment):
    history, prep, training, cv, extension, models = bootstrap(run)
    plan = read(run / 'run_plan.json')
    history.initialize(str(run / experiment))
    args = history.args_for(plan['parameters'], 40, 12)
    for key in PARAMETERS:
        value = getattr(args, key)
        assert value == plan['reference_config'][key], (key, value)
    args.output = run / experiment
    args.experiment = experiment
    args.rank_metric = 'energy_cosine' if experiment == 'energy_similarity' else 'peak'
    args.input_resampling = resampling(experiment)
    args.training_implementation = 'historical max_cv_search.py fit_fold'
    args.optimizer_implementation = 'AdamW(fused=True)'
    args.batch_randomness = 'torch.randperm(training_points, device=cuda), original global CUDA RNG'
    args.evaluation_batch_size = 32
    args.cv_seed = 42
    adapt_variants(history, training, models, experiment, run)
    return history, prep, training, cv, extension, args


def train_worker(run, experiment):
    import numpy as np
    import torch
    history, prep, training, cv, extension, args = initialize_worker(run, experiment)
    directory = run / experiment
    save(directory / 'config.json', training.serializable_args(args))
    oof = np.full(len(history.STATE['data']['label']), np.nan)
    results, reproduction = [], []
    for fold, (_, validation_rows) in enumerate(history.STATE['folds'], 1):
        print(f'[FOLD {fold}/5] {experiment}', flush=True)
        destination = directory / f'fold_{fold}'
        done = history.fit_fold(args, fold, destination)
        results.append(done)
        probabilities = np.load(destination / 'validation_probabilities.npy')
        oof[validation_rows] = probabilities
        if experiment == 'fft_reference':
            original = Path(read(run / 'run_plan.json')['historical_run']) / f'screen/random_0130/fold_{fold}'
            reference_p = np.load(original / 'validation_probabilities.npy')
            a = torch.load(destination / 'best.pt', map_location='cpu', weights_only=True)['model']
            b = torch.load(original / 'best.pt', map_location='cpu', weights_only=True)['model']
            item = {'fold': fold, 'probabilities_exact': bool(np.array_equal(probabilities, reference_p)),
                'max_probability_error': float(np.max(np.abs(probabilities - reference_p))),
                'model_tensors_exact': all(torch.equal(a[key], b[key]) for key in a),
                'best_epoch': done['best_epoch'], 'historical_best_epoch': read(original / 'complete.json')['best_epoch']}
            reproduction.append(item)
            save(directory / 'reproduction.json', reproduction)
            # Stop before the four experiments if the original training cannot be reproduced.
            if not (item['probabilities_exact'] and item['model_tensors_exact']):
                raise RuntimeError(f'Historical reproduction mismatch: {item}')
            del a, b
        print(f'[FOLD DONE] best_epoch={done["best_epoch"]} epochs={done["epochs_run"]} AUC={done["metrics"]["roc_auc"]:.4f}', flush=True)
        torch.cuda.empty_cache()
    np.save(directory / 'oof_probabilities.npy', oof)
    training.export_predictions(directory / 'oof_predictions.csv', history.STATE['data'], oof, .5)
    result = {'experiment': experiment, 'folds': results,
        'oof_metrics': training.compute_metrics(history.STATE['data']['label'], oof, .5),
        'auc_mean': float(np.mean([r['metrics']['roc_auc'] for r in results])),
        'auc_std': float(np.std([r['metrics']['roc_auc'] for r in results]))}
    save(directory / 'cv_results.json', result)
    save(directory / 'frozen_selection.json', {'threshold': .5, 'folds': 5, 'seeds': [2023],
        'checkpoints': {f'fold_{fold}/best.pt': sha(directory / f'fold_{fold}/best.pt') for fold in range(1, 6)},
        'fold_plan_sha256': sha(directory / 'fold_plan.json'), 'config_sha256': sha(directory / 'config.json'),
        'frozen_at_utc': now(), 'test_used_for_selection': False})
    print('[TRAIN DONE]', experiment, result['oof_metrics'], flush=True)


def evaluate_worker(run, experiment):
    import numpy as np
    import torch
    history, prep, training, cv, extension, args = initialize_worker(run, experiment)
    directory = run / experiment
    frozen = read(directory / 'frozen_selection.json')
    assert frozen['config_sha256'] == sha(directory / 'config.json')
    for name in JOBS:
        assert (run / name / 'frozen_selection.json').exists(), 'All models must be frozen first'
    test = inputs(prep, extension, args.dataset_root, 'test', experiment)
    save(directory / 'test_input.json', {'point_ids': test['point_ids'], 'resampling': resampling(experiment)})
    fold_probs, fold_beams = [], []
    for fold in range(1, 6):
        relative = f'fold_{fold}/best.pt'
        assert sha(directory / relative) == frozen['checkpoints'][relative]
        model, _ = history.setup_model(args)
        model.load_state_dict(torch.load(directory / relative, map_location='cuda', weights_only=True)['model'])
        regions = ('main', 'tail', 'whole') if experiment in ('six_channel', 'six_channel_beam_vote') else ('main', 'tail')
        probabilities, beams = [], []
        for start in range(0, len(test['label']), 32):
            tensors = [torch.as_tensor(test[key][start:start+32], device='cuda') for key in regions]
            probabilities.append(history.gpu_predict(model, *tensors))
            if experiment in ('beam_vote', 'six_channel_beam_vote'):
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = (model(tensors[0], tensors[1], whole_signal=tensors[2], return_beam_logits=True).float() if experiment == 'six_channel_beam_vote' else model(*tensors, return_beam_logits=True).float())
                beams.append(logits.softmax(-1)[:, :, 1].cpu().numpy())
        fold_probs.append(np.concatenate(probabilities))
        if beams:
            fold_beams.append(np.concatenate(beams))
        del model
        torch.cuda.empty_cache()
    by_fold = np.asarray(fold_probs)
    mean = by_fold.mean(0)
    np.save(directory / 'test_probabilities_by_fold.npy', by_fold)
    training.export_predictions(directory / 'test_predictions.csv', test, mean, .5)
    if fold_beams:
        beam_array = np.asarray(fold_beams)
        np.save(directory / 'test_beam_probabilities_by_fold.npy', beam_array)
        np.testing.assert_allclose(mean, beam_array.mean(0).mean(1), atol=1e-6, rtol=0)
    result = {'metrics': training.compute_metrics(test['label'], mean, .5),
        'selection_sha256': sha(directory / 'frozen_selection.json'), 'test_used_for_selection': False}
    if experiment == 'fft_reference':
        original = Path(read(run / 'run_plan.json')['historical_run']) / 'all_config_tests/20260928_234530/screen/random_0130'
        reference = np.load(original / 'test_probabilities_by_fold.npy')
        np.testing.assert_array_equal(by_fold, reference)
        assert result['metrics'] == read(original / 'result.json')['metrics']
        result['historical_test_probabilities_exact'] = True
    save(directory / 'test_result.json', result)
    print('[TEST]', experiment, result['metrics'], flush=True)


def audit_worker(run):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    plan = read(run / 'run_plan.json')
    for name, digest in plan['frozen_files'].items():
        assert sha(run / name) == digest, name
    for name, digest in plan['source_data_sha256'].items():
        assert sha(Path(plan['dataset_root']) / name) == digest, name
    rows = []
    for experiment in JOBS:
        directory = run / experiment
        frozen = read(directory / 'frozen_selection.json')
        for name, digest in frozen['checkpoints'].items():
            assert sha(directory / name) == digest, name
        cv = read(directory / 'cv_results.json')
        test = read(directory / 'test_result.json')
        for filename, expected, n in [('oof_predictions.csv', cv['oof_metrics'], 201),
                                      ('test_predictions.csv', test['metrics'], 67)]:
            with (directory / filename).open(encoding='utf-8-sig', newline='') as stream:
                values = list(csv.DictReader(stream))
            assert len(values) == len({v['point_id'] for v in values}) == n
            y = np.array([int(v['true_label']) for v in values])
            p = np.array([float(v['probability_below_boundary']) for v in values])
            prediction = (p >= .5).astype(int)
            for name, value in [('accuracy', accuracy_score(y, prediction)),
                                ('balanced_accuracy', balanced_accuracy_score(y, prediction)),
                                ('roc_auc', roc_auc_score(y, p))]:
                assert abs(value - expected[name]) < 1e-12, (experiment, filename, name)
            assert confusion_matrix(y, prediction, labels=[0, 1]).tolist() == expected['confusion_matrix']
        rows.append({'experiment': experiment, 'oof_accuracy': cv['oof_metrics']['accuracy'],
            'fold_auc_mean': cv['auc_mean'], 'test_accuracy': test['metrics']['accuracy'],
            'test_correct': test['metrics']['correct'], 'test_auc': test['metrics']['roc_auc'],
            'test_balanced_accuracy': test['metrics']['balanced_accuracy']})
    assert max((run / e / 'frozen_selection.json').stat().st_mtime_ns for e in JOBS) < min(
        (run / e / 'test_input.json').stat().st_mtime_ns for e in JOBS)
    assert read(run / 'fft_reference/test_result.json')['historical_test_probabilities_exact']
    save(run / 'comparison.json', rows)
    with (run / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(run / 'result_audit.json', {'passed': True, 'models_verified': 25,
        'historical_reference_models_and_predictions_exact': True,
        'source_code_and_data_hashes_verified': True, 'csv_metrics_recomputed': True,
        'all_models_frozen_before_test': True, 'threshold': .5,
        'baseline_only_changes_resampling': True, 'comparison': rows})
    print('[AUDIT PASS]', json.dumps(rows, ensure_ascii=False), flush=True)


def create_run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(HISTORY / 'manifest.json')
    snapshot = output / 'code_snapshot'
    for name, digest in manifest['code'].items():
        source = HISTORY / 'code_snapshot' / name
        assert sha(source) == digest, name
        destination = snapshot / 'historical' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name, digest in manifest['cache'].items():
        assert sha(HISTORY / name) == digest, name
    shutil.copy2(ROOT / 'prepare_data.py', snapshot / 'experiment_prepare_data.py')
    shutil.copy2(ROOT / 'models/vit_1d_cls.py', snapshot / 'experiment_models.py')
    shutil.copy2(Path(__file__), snapshot / 'controlled_driver.py')
    reference = read(HISTORY / 'screen/random_0130/config.json')
    current = load_module('current_defaults_for_audit', ROOT / 'configs.py')
    for key in PARAMETERS:
        value = getattr(current.CFGMaxFrame, key)
        assert (list(value) if isinstance(value, tuple) else value) == reference[key], key
    dataset = Path(reference['dataset_root'])
    for name, digest in manifest['source'].items():
        assert sha(dataset / name) == digest, name
    plan = {'created_at_utc': now(), 'historical_run': str(HISTORY), 'dataset_root': str(dataset),
        'reference_config': reference, 'parameters': read(HISTORY / 'screen/random_0130/result.json')['parameters'],
        'historical_code_sha256': manifest['code'], 'source_data_sha256': manifest['source'],
        'requested_experiments': list(EXPERIMENTS), 'reference_models': 5, 'experiment_models': 20,
        'precision': 'bf16', 'optimizer': 'AdamW(fused=True)', 'shuffle': 'CUDA torch.randperm',
        'attention': 'historical SDPA automatic backend; MHA fastpath disabled',
        'test_batch_size': 32, 'threshold': .5, 'python': sys.executable}
    save(output / 'run_plan.json', plan)
    status(output, phase='preparing')
    dispatch(output, 'prepare')
    status(output, phase='prepared', run_plan_sha256=sha(output / 'run_plan.json'))
    return output


def dispatch(run, stage, experiment=None):
    label = f'{experiment}_{stage}' if experiment else stage
    log_path = run / f'{label}.log'
    command = [sys.executable, '-X', 'utf8', '-B', '-u', str(run / 'code_snapshot/controlled_driver.py'),
               '--worker', stage, '--run', str(run)]
    if experiment:
        command += ['--experiment', experiment]
    print('[START]', label, flush=True)
    with log_path.open('w', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=run / 'code_snapshot', stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        status(run, child_pid=child.pid, current_log=str(log_path), current_stage=stage, current_experiment=experiment)
        code = child.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; see {log_path}')


def execute(run):
    run = run.resolve()
    state = read(run / 'status.json')
    assert state['phase'] == 'prepared', 'Only a new prepared run can be started'
    assert sha(run / 'run_plan.json') == state['run_plan_sha256']
    for name, digest in read(run / 'run_plan.json')['frozen_files'].items():
        assert sha(run / name) == digest, name
    status(run, phase='training', started_at_utc=now(), runner_pid=os.getpid(), completed_training=[], completed_evaluation=[])
    try:
        for stage in ('train', 'evaluate'):
            finished = []
            for experiment in JOBS:
                dispatch(run, stage, experiment)
                finished.append(experiment)
                status(run, **{('completed_training' if stage == 'train' else 'completed_evaluation'): finished})
        dispatch(run, 'audit')
        status(run, phase='complete', finished_at_utc=now(), child_pid=None, current_experiment=None)
        print('[DONE]', run, flush=True)
    except BaseException as error:
        status(run, phase='failed', error=str(error), failed_at_utc=now())
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--worker', choices=('prepare', 'train', 'evaluate', 'audit'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--experiment', choices=JOBS)
    args = parser.parse_args()
    if args.worker:
        functions = {'prepare': prepare_worker, 'train': train_worker, 'evaluate': evaluate_worker, 'audit': audit_worker}
        functions[args.worker](args.run, args.experiment) if args.worker in ('train', 'evaluate') else functions[args.worker](args.run)
        return
    if args.prepared:
        execute(args.prepared)
        return
    output = args.output or ROOT / 'results' / ('controlled_random0130_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    run = create_run(output)
    print('[PREPARED]', run, flush=True)
    if not args.prepare_only:
        execute(run)


if __name__ == '__main__':
    main()
