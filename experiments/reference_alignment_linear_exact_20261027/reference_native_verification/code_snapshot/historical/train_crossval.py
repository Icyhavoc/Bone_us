"""合并train/validation做点位级分层五折；固定test只在全部模型冻结后评估。"""
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from prepare_data import ROOT, audit_dataset, make_input, input_metadata, save_json, sha256
from train_minimal import (get_args, Trainer, Signalset, fetch_model, smoke_test,
                           compute_metrics, export_predictions, serializable_args)

POINT_ARRAYS = ('main', 'tail', 'main_frame_ids', 'tail_frame_ids',
                'selected_frame_indices', 'label', 'indices')


def subset(data, rows):
    rows = np.asarray(rows, dtype=np.int64)
    result = dict(data)
    for key in POINT_ARRAYS:
        if key in data:
            result[key] = data[key][rows]
    result['point_ids'] = [data['point_ids'][i] for i in rows]
    return result


def merge_development(train, validation):
    """按源点位索引排序；帧轴保持完整，双分支与选帧索引同时切分。"""
    for key in ('input_index', 'input_name', 'dataset_root', 'rank_metric', 'boundary_mm'):
        if train[key] != validation[key]:
            raise ValueError(f'训练和验证输入的{key}不一致')
    data = dict(train)
    for key in POINT_ARRAYS:
        if key in train:
            data[key] = np.concatenate([train[key], validation[key]], axis=0)
    data['point_ids'] = train['point_ids'] + validation['point_ids']
    if len(set(data['point_ids'])) != len(data['point_ids']) or len(np.unique(data['indices'])) != len(data['indices']):
        raise ValueError('合并输入包含重复点位或源索引')
    return subset(data, np.argsort(data['indices'], kind='stable'))


def make_folds(data, seed, heldout_ids, heldout_indices):
    n = len(data['label'])
    if set(data['point_ids']) & set(heldout_ids) or set(data['indices']) & set(heldout_indices):
        raise ValueError('开发集与独立test存在点位交叉')
    if np.any(np.bincount(data['label'], minlength=2) < 5):
        raise ValueError('每类至少需要5个点位才能分层五折')
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n), data['label']))
    seen = np.zeros(n, dtype=np.int64)
    for train_rows, val_rows in folds:
        if set(train_rows) & set(val_rows) or len(set(train_rows) | set(val_rows)) != n:
            raise ValueError('折内训练与验证划分异常')
        seen[val_rows] += 1
    if not np.all(seen == 1):
        raise ValueError('每个开发点位必须恰好作为验证样本一次')
    return folds


def fold_plan(data, folds, args, heldout_ids):
    records = []
    for fold, (tr, va) in enumerate(folds, 1):
        train_groups = {data['point_ids'][i].split('_')[0] for i in tr}
        val_groups = {data['point_ids'][i].split('_')[0] for i in va}
        records.append(dict(fold=fold, train_points=len(tr), validation_points=len(va),
            train_model_label_counts=np.bincount(data['label'][tr], minlength=2).tolist(),
            validation_model_label_counts=np.bincount(data['label'][va], minlength=2).tolist(),
            train_development_indices=tr.tolist(), validation_development_indices=va.tolist(),
            train_source_indices=data['indices'][tr].tolist(), validation_source_indices=data['indices'][va].tolist(),
            train_point_ids=[data['point_ids'][i] for i in tr],
            validation_point_ids=[data['point_ids'][i] for i in va],
            shared_n_groups=sorted(train_groups & val_groups)))
    return dict(protocol='train_plus_validation_stratified_5fold', cv_seed=args.cv_seed,
        group_unit='point', group_by_bone=False, development_points=len(data['label']),
        development_source_indices=data['indices'].tolist(), development_point_ids=data['point_ids'],
        development_model_label_counts=np.bincount(data['label'], minlength=2).tolist(),
        heldout_test_points=len(heldout_ids), heldout_test_point_ids=heldout_ids,
        model_seeds=args.seeds, total_models=5*len(args.seeds), decision_threshold=0.5,
        checkpoint_selection='fold_validation_auc; early_stopping_within_each_fold',
        evaluation_note='OOF validation also selects epoch; not nested CV or bone-grouped evaluation',
        input_window=args.input_window, input_scaling='none', folds=records)


def predict_checkpoint(args, data, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    dataset = Signalset(data, checkpoint['scaling'])
    model = fetch_model(args).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False):
            logits = model(batch['main'].to(args.device), batch['tail'].to(args.device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError('测试预测不是有限值')
            probabilities.append(logits.softmax(dim=1)[:,1].cpu().numpy())
    del model
    return np.concatenate(probabilities)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cv-seed', type=int, default=42)
    parser.add_argument('--preview-only', action='store_true')
    extra, remaining = parser.parse_known_args(argv)
    # 默认恰好5个模型；显式--seeds可指定每折多个随机种子。
    if not any(arg == '--seeds' or arg.startswith('--seeds=') for arg in remaining):
        from configs import get_cfg, INPUT_INDEX
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument('--input-index', type=int, default=INPUT_INDEX)
        selected, _ = selector.parse_known_args(remaining)
        remaining += ['--seeds', str(get_cfg(selected.input_index).seeds[0])]
    args = get_args(remaining)
    if not 0 <= extra.cv_seed < 2**32:
        raise ValueError('cv-seed须处于[0,2**32)')
    args.cv_seed = extra.cv_seed
    args.preview_only = extra.preview_only
    args.folds = 5
    args.protocol = 'train_plus_validation_stratified_5fold'
    if not any(arg == '--output' or arg.startswith('--output=') for arg in remaining):
        parent = 'verification' if args.smoke_test or args.preview_only else 'results'
        args.output = ROOT/parent/'cv5'/args.dataset_info['dataset_tag']/args.input_name/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return args


def run(args):
    torch.set_num_threads(4)
    audit = audit_dataset(args.dataset_root, require_source_signal=args.input_index in (2,3))
    train = make_input(args.dataset_root, 'train', args.seq_len, args.input_index, args.rank_metric)
    validation = make_input(args.dataset_root, 'validation', args.seq_len, args.input_index, args.rank_metric)
    development = merge_development(train, validation)
    del train, validation
    heldout_ids = np.load(args.dataset_root/'test/arrays/point_ids.npy', allow_pickle=False).tolist()
    heldout_indices = np.load(args.dataset_root/'test/indices.npy', allow_pickle=False)
    folds = make_folds(development, args.cv_seed, heldout_ids, heldout_indices)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'config.json', serializable_args(args))
    save_json(args.output/'data_audit.json', audit)
    save_json(args.output/'development_input.json', input_metadata(development))
    plan = fold_plan(development, folds, args, heldout_ids)
    save_json(args.output/'fold_plan.json', plan)
    np.savez(args.output/'fold_indices.npz', **{
        f'fold{fold}_{part}': rows for fold,pair in enumerate(folds,1)
        for part,rows in zip(('train','validation'),pair)})
    source_hashes = {p.relative_to(args.dataset_root).as_posix():sha256(p)
                     for p in sorted(args.dataset_root.rglob('*')) if p.is_file()}
    code_hashes = {p.relative_to(ROOT).as_posix():sha256(p) for p in
        [Path(__file__),ROOT/'train_minimal.py',ROOT/'prepare_data.py',ROOT/'configs.py',ROOT/'models/vit_1d_cls.py']}
    save_json(args.output/'input_code_manifest.json', dict(dataset_sha256=source_hashes, code_sha256=code_hashes))
    print(f'[CV5] development={len(development["label"])} test={len(heldout_ids)} models={5*len(args.seeds)}', flush=True)
    if args.preview_only:
        print('[PREVIEW]', args.output, flush=True)
        return
    oof = np.full((len(development['label']),len(args.seeds)), np.nan, dtype=np.float64)
    fold_metrics, checkpoints = [], {}
    for fold,(tr,va) in enumerate(folds,1):
        train_data, val_data = subset(development,tr), subset(development,va)
        for j,seed in enumerate(args.seeds):
            destination = args.output/f'fold_{fold}'/f'seed_{seed}'
            print(f'[FOLD {fold}/5] seed={seed} train={len(tr)} validation={len(va)}', flush=True)
            trainer = Trainer(args,train_data,val_data,seed,destination)
            if args.smoke_test:
                smoke_test(trainer)
            else:
                probabilities = trainer.train()
                if probabilities.shape != (len(va),) or not np.isfinite(probabilities).all():
                    raise ValueError('折验证预测形状或数值异常')
                oof[va,j] = probabilities
                checkpoint = destination/'best.pt'
                checkpoints[checkpoint.relative_to(args.output).as_posix()] = sha256(checkpoint)
            del trainer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not args.smoke_test:
            probabilities = oof[va].mean(axis=1)
            metrics = compute_metrics(val_data['label'],probabilities,0.5)
            fold_metrics.append({'fold':fold, **metrics})
            save_json(args.output/f'fold_{fold}'/'metrics.json',metrics)
            export_predictions(args.output/f'fold_{fold}'/'validation_predictions.csv',val_data,probabilities,0.5)
    if args.smoke_test:
        save_json(args.output/'smoke_summary.json',dict(status='passed',folds=5,model_seeds=args.seeds,
            all_folds_forward_backward_and_checkpoint=True, test_inference_performed=False))
        print('[SMOKE DONE]',args.output,flush=True)
        return
    if not np.isfinite(oof).all(): raise ValueError('OOF预测不完整')
    probabilities = oof.mean(axis=1)
    np.save(args.output/'oof_probabilities_by_seed.npy',oof)
    export_predictions(args.output/'oof_predictions.csv',development,probabilities,0.5)
    summary = dict(oof_metrics=compute_metrics(development['label'],probabilities,0.5),
        folds=fold_metrics,decision_threshold=0.5,model_seeds=args.seeds,
        interpretation='fold-held-out predictions; same fold validation used for early stopping; not nested CV')
    for metric in ('accuracy','balanced_accuracy','macro_f1','roc_auc'):
        values = np.array([r[metric] for r in fold_metrics])
        summary[metric+'_mean_std'] = {'mean':float(values.mean()),'std_ddof0':float(values.std())}
    save_json(args.output/'cv_results.json',summary)
    # 在读取测试模型输入前冻结全部折模型，不依据验证排名筛选折。
    frozen = dict(method='equal_probability_mean_of_all_folds_and_seeds', threshold=0.5,
        threshold_selection='fixed_a_priori',checkpoints=checkpoints,seeds=args.seeds,folds=5,
        fold_plan_sha256=sha256(args.output/'fold_plan.json'),dataset_info=args.dataset_info,
        input_window=args.input_window,input_scaling='none')
    save_json(args.output/'frozen_selection.json',frozen)
    for relative,digest in source_hashes.items():
        if sha256(args.dataset_root/relative) != digest: raise RuntimeError('训练期间源数据变化')
    if args.skip_test:
        print('[DONE] CV完成，未执行测试集推理。',args.output,flush=True)
        return
    test_data = make_input(args.dataset_root,'test',args.seq_len,args.input_index,args.rank_metric)
    save_json(args.output/'test_input.json',input_metadata(test_data))
    predictions=[]
    for relative,digest in checkpoints.items():
        path=args.output/relative
        if sha256(path) != digest: raise RuntimeError('冻结后权重变化')
        predictions.append(predict_checkpoint(args,test_data,path))
    test_probabilities=np.mean(predictions,axis=0)
    result=compute_metrics(test_data['label'],test_probabilities,0.5)
    export_predictions(args.output/'test_predictions.csv',test_data,test_probabilities,0.5)
    save_json(args.output/'test_result.json',dict(metrics=result,
        selection_sha256=sha256(args.output/'frozen_selection.json'),test_used_for_selection=False,
        scope='existing point-level internal holdout; previously inspected; not bone-grouped'))
    print(f'[TEST] accuracy={result["accuracy"]:.4f} ({result["correct"]}/{result["n"]}) AUC={result["roc_auc"]:.4f}',flush=True)
    print('[DONE]',args.output,flush=True)


if __name__ == '__main__':
    run(parse_args())
