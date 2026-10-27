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
    x=[torch.as_tensor(data[k][tr],device='cuda') for k in active_regions(args)]
    vx=[torch.as_tensor(data[k][va],device='cuda') for k in active_regions(args)]
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
    criterion=controlled_criterion(args)
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
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=criterion(controlled_logits(model,x,rows,args),y[rows])
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
