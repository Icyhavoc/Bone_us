"""Max-frame four-ViT search: >=256 complete CV configurations AND >2h active training.

Run from the frozen snapshot. No test predictions until the final selection is frozen.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import concurrent.futures as cf
import contextlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPACE = dict(patch_size=[4,8,16,32,64], dim=[32,64,96,128,256], depth=[1,2,3,4],
    heads=[2,4,8,16], mlp_ratio=[2,4], lr=[3e-5,1e-4,3e-4,1e-3],
    dropout=[0,.1,.2,.25,.3,.4], weight_decay=[0,.001,.01,.02,.05,.1], batch_size=[4,8,16,24,32])
BASELINE = dict(patch_size=8,dim=128,depth=3,heads=16,mlp_dim=256,lr=1e-4,
    dropout=.25,weight_decay=.02,batch_size=24,label_smoothing=0.,warmup=5)
STATE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def draw_candidates(count):
    rng = random.Random(20260928)
    result, seen = [dict(id='baseline',parameters=BASELINE)], {json.dumps(BASELINE,sort_keys=True)}
    while len(result) < count:
        p = {key:rng.choice(values) for key,values in SPACE.items()}
        if p['dim'] % p['heads'] or p['dim']//p['heads'] < 8:
            continue
        p['mlp_dim'] = p['dim']*p.pop('mlp_ratio')
        p.update(label_smoothing=0.,warmup=5)
        key = json.dumps(p,sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(dict(id=f'random_{len(result):04d}',parameters=p))
    return result


def initialize(run):
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run requires CUDA and BF16')
    root = Path(run)
    data = read(root/'development.json')
    with np.load(root/'development.npz',allow_pickle=False) as arrays:
        data.update({key:arrays[key] for key in arrays.files})
    with np.load(root/'fold_indices.npz',allow_pickle=False) as arrays:
        folds = [(arrays[f'fold{i}_train'],arrays[f'fold{i}_validation']) for i in range(1,6)]
    STATE.update(run=root,data=data,folds=folds)


def event(kind, **kwargs):
    path = STATE['run']/'events'/f'{os.getpid()}.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(kind=kind,time=time.time(),**kwargs))+'\n')


def active_seconds(run, live=True):
    intervals = []
    for path in (Path(run)/'events').glob('*.jsonl'):
        start, last = None, None
        for line in path.read_text(encoding='utf-8').splitlines():
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            if item['kind']=='start':
                stage=item.get('stage',Path(item['path']).parent.parent.name if 'path' in item else 'screen')
                start=last=item['time'] if stage=='screen' else None
            elif item['kind']=='heartbeat' and start is not None: last=item['time']
            elif item['kind'] in ('end','interrupted') and start is not None:
                intervals.append((start,item['time'])); start=None
        if start is not None:
            intervals.append((start,time.time() if live else last))
    intervals.sort()
    total, end = 0., -math.inf
    for a,b in intervals:
        total += max(0.,b-max(a,end)); end=max(end,b)
    return total


def close_stale_events(run):
    for path in (Path(run)/'events').glob('*.jsonl'):
        start, last = None, None
        for line in path.read_text(encoding='utf-8').splitlines():
            try: item=json.loads(line)
            except json.JSONDecodeError: continue
            if item['kind']=='start': start=last=item['time']
            elif item['kind']=='heartbeat' and start is not None: last=item['time']
            elif item['kind'] in ('end','interrupted'): start=None
        if start is not None:
            with path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(kind='interrupted',time=last))+'\n')


def args_for(parameters, epochs, patience):
    from train_minimal import get_args
    opts=['--input-index','2','--dataset-root',read(STATE['run']/'protocol.json')['dataset_root'],
          '--device','cuda','--seeds','2023','--epochs',str(epochs),'--patience',str(patience)]
    for key,value in parameters.items(): opts += ['--'+key.replace('_','-'),str(value)]
    args=get_args(opts)
    assert args.split_beams and args.label_smoothing==0
    args.precision='bf16'; args.attention_backend='pytorch_sdpa_auto'
    return args


def setup_model(args):
    import torch
    from train_minimal import fetch_model,set_seed
    set_seed(2023)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    model=fetch_model(args).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,fused=True)
    return model,optimizer


def gpu_predict(model, main, tail):
    import torch
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(main,tail).float()
    if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite logits')
    return logits.softmax(1)[:,1].cpu().numpy()


def atomic_torch_save(value,path):
    import torch
    temp=path.with_suffix('.tmp')
    torch.save(value,temp); os.replace(temp,path)


def fit_fold(args, fold, destination):
    import numpy as np
    import torch
    from train_minimal import compute_metrics,serializable_args,export_predictions
    from train_crossval import subset
    from prepare_data import sha256
    destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'complete.json'
    if marker.exists():
        done=read(marker)
        if sha256(destination/'best.pt')!=done['checkpoint_sha256']: raise RuntimeError('checkpoint hash mismatch')
        if sha256(destination/'validation_probabilities.npy')!=done['predictions_sha256']: raise RuntimeError('prediction hash mismatch')
        return done
    tr,va=STATE['folds'][fold-1]
    data=STATE['data']; valid=subset(data,va)
    x=[torch.as_tensor(data[k][tr],device='cuda') for k in ('main','tail')]
    vx=[torch.as_tensor(data[k][va],device='cuda') for k in ('main','tail')]
    y=torch.as_tensor(data['label'][tr],device='cuda')
    model,optim=setup_model(args)
    history=[]; best=-1.; best_epoch=0; stale=0; first=1
    resume=destination/'resume.pt'
    if resume.exists():
        # Locally generated optimizer/RNG state, never an external checkpoint.
        checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']); optim.load_state_dict(checkpoint['optimizer'])
        history=checkpoint['history']; best=checkpoint['best']; best_epoch=checkpoint['best_epoch']; stale=checkpoint['stale']
        first=checkpoint['epoch']+1
        torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        # Restore the best matching the committed epoch, even if a later partial epoch wrote best.pt.
        atomic_torch_save(checkpoint['best_checkpoint'],destination/'best.pt')
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.)
    started=time.time(); torch.cuda.reset_peak_memory_stats()
    event('start',fold=fold,path=str(destination),stage=destination.parent.parent.name)
    try:
        for epoch in range(first,args.epochs+1):
            if stale>=args.patience: break
            model.train(); total=0.
            factor=epoch/max(1,args.warmup) if epoch<=args.warmup else .5*(1+np.cos(np.pi*(epoch-args.warmup)/max(1,args.epochs-args.warmup)))
            for group in optim.param_groups: group['lr']=args.lr*factor
            order=torch.randperm(len(tr),device='cuda')
            for rows in order.split(args.batch_size):
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=criterion(model(x[0][rows],x[1][rows]),y[rows])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optim.step(); total+=loss.item()*len(rows)
            probabilities=gpu_predict(model,*vx)
            metrics=compute_metrics(valid['label'],probabilities,.5)
            if metrics['roc_auc']>best+1e-8:
                best=metrics['roc_auc']; best_epoch=epoch; stale=0
                atomic_torch_save(dict(model=model.state_dict(),scaling=None,seed=2023,epoch=epoch,
                    validation_auc=best,config=serializable_args(args)),destination/'best.pt')
            else: stale+=1
            history.append(dict(epoch=epoch,train_loss=total/len(tr),validation_auc=metrics['roc_auc'],
                validation_accuracy_05=metrics['accuracy'],lr=args.lr*factor))
            save(destination/'history.json',history)
            if epoch%5==0 or stale>=args.patience or epoch==args.epochs:
                atomic_torch_save(dict(model=model.state_dict(),optimizer=optim.state_dict(),history=history,
                    best=best,best_epoch=best_epoch,stale=stale,epoch=epoch,
                    best_checkpoint=torch.load(destination/'best.pt',map_location='cpu',weights_only=True),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),resume)
            event('heartbeat',fold=fold,epoch=epoch)
            save(STATE['run']/'workers'/f'{os.getpid()}.json',dict(phase='training',path=str(destination),fold=fold,
                epoch=epoch,best_auc=best,time=time.time(),peak_mb=torch.cuda.max_memory_allocated()/2**20))
        checkpoint=torch.load(destination/'best.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(checkpoint['model'])
        probabilities=gpu_predict(model,*vx)
        np.save(destination/'validation_probabilities.npy',probabilities)
        export_predictions(destination/'validation_predictions.csv',valid,probabilities,.5)
        done=dict(fold=fold,best_epoch=best_epoch,epochs_run=len(history),metrics=compute_metrics(valid['label'],probabilities,.5),
            parameter_count=sum(p.numel() for p in model.parameters()),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            attempt_seconds=time.time()-started,checkpoint_sha256=sha256(destination/'best.pt'),
            predictions_sha256=sha256(destination/'validation_probabilities.npy'))
        save(marker,done)
        # Only remove this run's redundant restart checkpoint after a verified complete marker.
        if resume.exists(): resume.unlink()
        return done
    finally:
        event('end',fold=fold,path=str(destination))
        save(STATE['run']/'workers'/f'{os.getpid()}.json',dict(phase='idle',time=time.time()))


def train_candidate(candidate, stage, smoke=False):
    import numpy as np
    import torch
    from train_minimal import compute_metrics,export_predictions,serializable_args
    root=STATE['run']/stage/candidate['id']; root.mkdir(parents=True,exist_ok=True)
    args=args_for(candidate['parameters'],1 if smoke else (40 if stage=='screen' else 120),12 if stage=='screen' else 20)
    if smoke: args.warmup=0
    save(root/'config.json',serializable_args(args))
    try:
        with (root/'training.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            done=[]; oof=np.full(len(STATE['data']['label']),np.nan)
            for fold,(_,va) in enumerate(STATE['folds'],1):
                info=fit_fold(args,fold,root/f'fold_{fold}'); done.append(info)
                oof[va]=np.load(root/f'fold_{fold}'/'validation_probabilities.npy')
                torch.cuda.empty_cache()
            if not np.isfinite(oof).all(): raise RuntimeError('incomplete OOF')
            np.save(root/'oof_probabilities.npy',oof)
            export_predictions(root/'oof_predictions.csv',STATE['data'],oof,.5)
            result=dict(id=candidate['id'],parameters=candidate['parameters'],folds=done,
                auc_mean=float(np.mean([r['metrics']['roc_auc'] for r in done])),
                auc_std=float(np.std([r['metrics']['roc_auc'] for r in done])),
                balanced_accuracy_mean=float(np.mean([r['metrics']['balanced_accuracy'] for r in done])),
                accuracy_mean=float(np.mean([r['metrics']['accuracy'] for r in done])),
                parameter_count=done[0]['parameter_count'],oof_metrics=compute_metrics(STATE['data']['label'],oof,.5))
            save(root/'result.json',result)
            return dict(ok=True,**result)
    except Exception as exc:
        torch.cuda.empty_cache()
        error=dict(ok=False,id=candidate['id'],oom=isinstance(exc,torch.OutOfMemoryError),error=traceback.format_exc())
        save(root/'error.json',error)
        return error


def benchmark(case, index):
    import torch
    args=args_for(BASELINE,40,12)
    model,optim=setup_model(args)
    main=torch.randn(24,2,512,device='cuda'); tail=torch.randn_like(main); y=torch.arange(24,device='cuda')%2
    model.eval()
    with torch.no_grad():
        fp=model(main,tail).float()
        with torch.autocast('cuda',dtype=torch.bfloat16): mixed=model(main,tail).float()
    error=(fp-mixed).abs().max().item()
    if error>.05 or not torch.isfinite(mixed).all(): raise RuntimeError(f'BF16 comparison failed: {error}')
    def step():
        model.train(); optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=torch.nn.functional.cross_entropy(model(main,tail),y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optim.step()
    for _ in range(4): step()
    assert all(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in b.parameters()) for b in model.branches)
    root=STATE['run']/'preflight'/case; root.mkdir(parents=True,exist_ok=True)
    save(root/f'ready_{index}.json',dict(pid=os.getpid(),max_abs_fp32_bf16=error))
    deadline=time.time()+120
    while not (root/'go').exists():
        if time.time()>deadline: raise TimeoutError('benchmark rendezvous')
        time.sleep(.02)
    torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for _ in range(30): step()
    torch.cuda.synchronize()
    result=dict(seconds=time.perf_counter()-start,steps=30,peak_mb=torch.cuda.max_memory_allocated()/2**20,fp32_bf16_error=error)
    save(root/f'result_{index}.json',result)
    del model,optim; torch.cuda.empty_cache()
    return result


def preflight(run, smoke):
    results=[]
    with cf.ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        for workers in (1,2,3):
            case=f'workers_{workers}_{int(time.time())}'
            jobs=[pool.submit(benchmark,case,i) for i in range(workers)]
            root=run/'preflight'/case; deadline=time.time()+150
            while len(list(root.glob('ready_*.json')))<workers:
                for job in jobs:
                    if job.done() and job.exception(): raise job.exception()
                if time.time()>deadline: raise TimeoutError('preflight ready timeout')
                time.sleep(.2)
            (root/'go').touch()
            measurements=[job.result() for job in jobs]
            result=dict(workers=workers,steps_per_second=30*workers/max(m['seconds'] for m in measurements),measurements=measurements)
            results.append(result); print('[PREFLIGHT]',json.dumps(result),flush=True)
    best=max(results,key=lambda r:r['steps_per_second'])
    # Avoid an extra process unless it gives at least 10% throughput benefit.
    chosen=min(r['workers'] for r in results if r['steps_per_second']>=best['steps_per_second']/1.1)
    save(run/'preflight.json',dict(results=results,selected_workers=chosen))
    return chosen


def rank_results(results):
    # Stable groups within 0.005 of the highest remaining mean AUC.
    pending=list(results); ranked=[]
    while pending:
        high=max(r['auc_mean'] for r in pending)
        tied=[r for r in pending if high-r['auc_mean']<.005]
        tied.sort(key=lambda r:(-r['balanced_accuracy_mean'],r['auc_std'],r['parameter_count'],r['id']))
        ranked.extend(tied); used={r['id'] for r in tied}; pending=[r for r in pending if r['id'] not in used]
    return ranked


def run_batch(run, candidates, stage, workers, smoke):
    existing=[]; todo=[]
    for c in candidates:
        path=run/stage/c['id']/'result.json'
        if path.exists(): existing.append(read(path))
        else: todo.append(c)
    failures=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        queue=list(todo); running={}; serial_only=False; last_print=0.
        while queue or running:
            while queue and len(running)<(1 if serial_only else workers):
                c=queue.pop(0); running[pool.submit(train_candidate,c,stage,smoke)]=c
            finished,_=cf.wait(running,timeout=2,return_when=cf.FIRST_COMPLETED)
            for future in finished:
                c=running.pop(future)
                try: r=future.result()
                except Exception: r=dict(ok=False,id=c['id'],oom=False,error=traceback.format_exc())
                if r['ok']:
                    existing.append(r)
                    print(f'[{stage}] {c["id"]} complete AUC={r["auc_mean"]:.4f} ({len(existing)}/{len(candidates)})',flush=True)
                else:
                    failures.append(dict(candidate=c,result=r))
                    if r.get('oom'): serial_only=True
                    print(f'[FAIL] {c["id"]} {r["error"][-800:]}',flush=True)
            if time.time()-last_print>=20:
                summary=dict(phase=stage,completed_in_batch=len(existing),batch_size=len(candidates),
                    running=[c['id'] for c in running.values()],pending=len(queue),failed=len(failures),
                    search_active_seconds=active_seconds(run) if stage=='screen' else read(run/'screen_summary.json')['active_seconds'],
                    workers=workers,time=time.time())
                save(run/'progress.json',summary); last_print=time.time()
                try:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip()
                    with (run/'gpu_usage.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(time=time.time(),gpu_percent_memory_mib_temperature_power=gpu))+'\n')
                except (OSError,subprocess.TimeoutExpired): pass
                ranked=rank_results(existing)
                save(run/f'{stage}_ranking.json',ranked)
    # Retry failed configurations alone, preserving any completed folds and epoch checkpoints.
    if failures:
        still=[]
        with cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
            for failure in failures:
                c=failure['candidate']; r=pool.submit(train_candidate,c,stage,smoke).result()
                if r['ok']: existing.append(r)
                else: still.append(dict(candidate=c,result=r))
        save(run/f'{stage}_failures_{int(time.time())}.json',still)
    ranked=rank_results(existing); save(run/f'{stage}_ranking.json',ranked)
    return ranked


def prepare(run,smoke):
    import numpy as np
    from configs import DATASET_ROOT
    from prepare_data import make_input,input_metadata,sha256,audit_dataset
    from train_crossval import merge_development,make_folds,fold_plan,parse_args
    if (run/'protocol.json').exists(): return
    run.mkdir(parents=True,exist_ok=True)
    snapshot=run/'code_snapshot'; snapshot.mkdir(exist_ok=True)
    files=['configs.py','prepare_data.py','train_minimal.py','train_crossval.py','models/vit_1d_cls.py','models/__init__.py','超参数探索/max_cv_search.py']
    for name in files:
        dest=snapshot/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(PROJECT/name,dest)
    save(run/'data_audit.json',audit_dataset(DATASET_ROOT,require_source_signal=True))
    train=make_input(DATASET_ROOT,'train',input_index=2); val=make_input(DATASET_ROOT,'validation',input_index=2)
    data=merge_development(train,val)
    ids=np.load(DATASET_ROOT/'test/arrays/point_ids.npy',allow_pickle=False).tolist()
    indices=np.load(DATASET_ROOT/'test/indices.npy',allow_pickle=False)
    folds=make_folds(data,42,ids,indices)
    arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    np.savez(run/'development.npz',**arrays); save(run/'development.json',{k:v for k,v in data.items() if k not in arrays})
    save(run/'development_input.json',input_metadata(data))
    args=parse_args(['--input-index','2']); save(run/'fold_plan.json',fold_plan(data,folds,args,ids))
    np.savez(run/'fold_indices.npz',**{f'fold{i}_{part}':rows for i,pair in enumerate(folds,1) for part,rows in zip(('train','validation'),pair)})
    save(run/'manifest.json',dict(source={str(p.relative_to(DATASET_ROOT)):sha256(p) for p in DATASET_ROOT.rglob('*') if p.is_file()},
        code={name:sha256(snapshot/name) for name in files},cache={name:sha256(run/name) for name in ('development.npz','development.json','fold_indices.npz','fold_plan.json')}))
    save(run/'protocol.json',dict(dataset_root=str(DATASET_ROOT),input_index=2,architecture='four_independent_region_beam_vit',
        min_candidates=2 if smoke else 256,min_active_seconds=0 if smoke else 7200,append_batch=16,search_seed=20260928,
        cv_seed=42,model_seed=2023,screen_epochs=40,screen_patience=12,review_epochs=120,review_patience=20,
        review_count=2 if smoke else 6,space=SPACE,baseline=BASELINE,precision='bf16',threshold=.5,smoke=smoke,
        time_rule='Union of active fold-training intervals; exclude preparation, preflight, idle and downtime',
        ranking_rule='mean fold AUC; within 0.005: balanced accuracy descending, AUC std ascending, parameter count ascending',
        interpretation='point-stratified CV used for model/epoch selection; not nested or bone-grouped'))
    save(run/'candidates.json',draw_candidates(2 if smoke else 256))


def verify_manifest(run):
    from prepare_data import sha256
    manifest=read(run/'manifest.json'); data=Path(read(run/'protocol.json')['dataset_root'])
    for category,root in [('code',run/'code_snapshot'),('cache',run),('source',data)]:
        for name,digest in manifest[category].items():
            if sha256(root/name)!=digest: raise RuntimeError(f'{category} changed: {name}')


def final_test(run,winner,smoke):
    import numpy as np
    import torch
    from prepare_data import make_input,sha256,input_metadata
    from train_minimal import compute_metrics,export_predictions
    paths={f'review/{winner["id"]}/fold_{fold}/best.pt':sha256(run/f'review/{winner["id"]}/fold_{fold}/best.pt') for fold in range(1,6)}
    frozen=dict(candidate=winner['id'],parameters=winner['parameters'],checkpoints=paths,threshold=.5,
        ensemble='equal probability mean of all five fold models',test_used_for_selection=False,
        fold_plan_sha256=sha256(run/'fold_plan.json'))
    if (run/'frozen_selection.json').exists():
        if read(run/'frozen_selection.json')!=frozen: raise RuntimeError('frozen selection changed')
    else: save(run/'frozen_selection.json',frozen)
    if smoke: return
    initialize(str(run))
    args=args_for(winner['parameters'],120,20)
    test=make_input(args.dataset_root,'test',input_index=2)
    save(run/'test_input.json',input_metadata(test))
    probs=[]
    for relative,digest in paths.items():
        if sha256(run/relative)!=digest: raise RuntimeError('frozen checkpoint changed')
        model,_=setup_model(args)
        checkpoint=torch.load(run/relative,map_location='cuda',weights_only=True); model.load_state_dict(checkpoint['model'])
        p=[]
        for i in range(0,len(test['label']),32):
            p.append(gpu_predict(model,*[torch.as_tensor(test[k][i:i+32],device='cuda') for k in ('main','tail')]))
        probs.append(np.concatenate(p)); del model; torch.cuda.empty_cache()
    probabilities=np.mean(probs,axis=0)
    np.save(run/'test_probabilities_by_fold.npy',np.asarray(probs))
    export_predictions(run/'test_predictions.csv',test,probabilities,.5)
    save(run/'test_result.json',dict(metrics=compute_metrics(test['label'],probabilities,.5),
        selection_sha256=sha256(run/'frozen_selection.json'),test_used_for_selection=False))


def report(run,winner):
    protocol=read(run/'protocol.json'); screen=read(run/'screen_summary.json')
    test=read(run/'test_result.json')['metrics'] if (run/'test_result.json').exists() else None
    lines=['# Max单帧四ViT搜索结果','',f'有效初筛配置：{screen["completed"]}；搜索训练活跃墙钟：{screen["active_seconds"]/3600:.3f}小时。',
        f'最佳配置：{winner["id"]}。每组固定五折，每折种子2023；每个模型含四个独立ViT。','',
        '| 参数 | 值 |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in winner['parameters'].items()]
    lines += ['',f'完整复核五折平均AUC：{winner["auc_mean"]:.6f} ± {winner["auc_std"]:.6f}。',
        f'平均准确率：{winner["accuracy_mean"]:.6f}；平均平衡准确率：{winner["balanced_accuracy_mean"]:.6f}。',
        f'OOF AUC：{winner["oof_metrics"]["roc_auc"]:.6f}。','']
    if test: lines += [f'测试集五折概率均值：AUC={test["roc_auc"]:.6f}，accuracy={test["accuracy"]:.6f}，balanced_accuracy={test["balanced_accuracy"]:.6f}。',f'混淆矩阵：{test["confusion_matrix"]}。','']
    lines += ['阈值固定0.5；模型1类表示厚度<1.3mm。输入512点、Tukey α=0.3、无RMS、label smoothing=0。',
        '点位级分层，未按骨编号分组；验证折也用于早停和参数选择，交叉验证分数不是独立无偏测试成绩。',
        '此前双ViT结果未参与本轮排名。']
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def execute(run):
    protocol=read(run/'protocol.json'); smoke=protocol['smoke']
    close_stale_events(run); verify_manifest(run)
    workers=read(run/'preflight.json')['selected_workers'] if (run/'preflight.json').exists() else preflight(run,smoke)
    print(f'[START] workers={workers} min_configs={protocol["min_candidates"]} min_seconds={protocol["min_active_seconds"]}',flush=True)
    candidates=read(run/'candidates.json')
    while True:
        ranked=run_batch(run,candidates,'screen',workers,smoke)
        seconds=active_seconds(run,live=False)
        if len(ranked)>=protocol['min_candidates'] and seconds>protocol['min_active_seconds']: break
        candidates=draw_candidates(len(candidates)+protocol['append_batch'])
        save(run/'candidates.json',candidates)
        print(f'[EXTEND] {len(candidates)} configurations; {seconds:.1f}s active',flush=True)
    save(run/'screen_summary.json',dict(completed=len(ranked),drawn=len(candidates),active_seconds=seconds,ranking=ranked))
    chosen=[next(c for c in candidates if c['id']=='baseline')]
    for row in ranked:
        if row['id']!='baseline': chosen.append(next(c for c in candidates if c['id']==row['id']))
        if len(chosen)>=protocol['review_count']: break
    save(run/'review_candidates.json',chosen)
    reviewed=run_batch(run,chosen,'review',workers,smoke)
    if len(reviewed)!=len(chosen): raise RuntimeError('Review has failed configurations; resume to retry before selection')
    winner=reviewed[0]; verify_manifest(run)
    final_test(run,winner,smoke); report(run,winner)
    save(run/'progress.json',dict(phase='complete',winner=winner['id'],screen_completed=len(ranked),
        search_active_seconds=seconds,review_completed=len(reviewed),test_evaluated=not smoke,time=time.time()))
    print('[COMPLETE]',run,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); run=args.run.resolve()
    if args.prepare:
        prepare(run,args.smoke); print('[PREPARED]',run,flush=True); return
    if PROJECT != (run/'code_snapshot').resolve(): raise RuntimeError('Execute the frozen code_snapshot script')
    # Windows kernel releases the byte-range lock if the scheduler exits or crashes.
    import msvcrt
    with (run/'scheduler.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise RuntimeError('This run already has an active scheduler')
        save(run/'scheduler.json',dict(pid=os.getpid(),started=time.time()))
        try:
            if (run/'progress.json').exists() and read(run/'progress.json').get('phase')=='complete':
                print('[ALREADY COMPLETE]',run,flush=True); return
            execute(run)
        except Exception:
            save(run/'fatal_error.json',dict(time=time.time(),error=traceback.format_exc()))
            print(traceback.format_exc(),flush=True); raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    main()
