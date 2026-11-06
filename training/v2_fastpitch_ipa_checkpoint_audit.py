"""Read-only CPU audit of versioned IPA FastPitch, retaining generic46checks.

Adds frozen phone/map/source bindings; original glyph-auditor source is unchanged.

Exit0: consistent finite checkpoint; exit1: stable failure; exit2: publication
changed during all retry attempts. The progress file is not a checkpoint.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))


class SnapshotChanged(RuntimeError):pass


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_output(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.chmod(0o600);os.replace(temporary,path)


def identity(stat):return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def compare_recipe(saved,current):
    """Operational metadata may differ; only three explicit resource knobs may change."""
    core={k:v for k,v in current.items() if k not in ('args','provenance')}
    resources={'batch_size','accumulation','precision'};changes={};other={}
    for key in sorted(set(saved)|set(core)):
        if key not in saved or key not in core or saved.get(key)!=core.get(key):
            value=dict(before=saved.get(key),after=core.get(key))
            (changes if key in resources else other)[key]=value
    return changes,other


def adam_layout(optimizer,parameters,model_state,step,torch,unused=()):
    """Reject missing/reset moments, shape changes, and incomplete parameter maps."""
    groups=optimizer.get('param_groups',[]);states=optimizer.get('state',{})
    ids=[i for group in groups for i in group.get('params',[])]
    mapping=(len(groups)==1 and ids==list(range(len(parameters))))
    required={i for i,(name,param) in enumerate(parameters) if param.requires_grad and name not in unused}
    all_ids=set(range(len(parameters)));present=set(states)
    shape_ok=dtype_ok=step_ok=True;errors=[]
    for i,state in states.items():
        if i not in all_ids:errors.append('unknown parameter ID '+str(i));shape_ok=False;continue
        name,param=parameters[i]
        if not isinstance(state,dict) or not {'step','exp_avg','exp_avg_sq'}<=set(state):
            errors.append(name+': incomplete Adam state');shape_ok=False;step_ok=False;continue
        for key in ('exp_avg','exp_avg_sq'):
            moment=state[key];good=torch.is_tensor(moment) and tuple(moment.shape)==tuple(param.shape)
            shape_ok=shape_ok and good
            dtype_ok=dtype_ok and torch.is_tensor(moment) and moment.dtype==model_state[name].dtype==param.dtype
            if not good:errors.append(name+': '+key+' shape mismatch')
        value=state['step']
        valid=(torch.is_tensor(value) and value.numel()==1 and bool(torch.isfinite(value).all())
               and float(value)>0 and (float(value)==step if i in required else float(value)<=step))
        step_ok=step_ok and valid
        if not valid:errors.append(name+': invalid Adam step')
    missing=[parameters[i][0] for i in sorted(required-present)]
    coverage=required<=present<=all_ids
    hyperparameters=(len(groups)==1 and all(tuple(g.get('betas',()))==(.9,.98) and g.get('eps')==1e-8
                     and g.get('weight_decay')==1e-6 and not g.get('amsgrad',False) for g in groups))
    checks=dict(optimizer_parameter_map_complete=mapping,optimizer_active_parameters_covered=coverage,
                optimizer_moment_shapes_valid=shape_ok,optimizer_moment_dtypes_valid=dtype_ok,
                optimizer_complete_steps_valid=step_ok,optimizer_adam_hyperparameters_unchanged=hyperparameters)
    return checks,dict(parameters=len(parameters),active_parameters=len(required),state_entries=len(states),
                       missing_active_parameters=missing,explicit_unused_parameters=sorted(unused),errors=errors)


def acoustic_config_matches(current_features, config, ipa_vocab):
    """The immutable cache stays graphemic; only model-input vocabulary differs."""
    return (current_features == config.get('raw_source_features')
            and config['features'] == dict(current_features, vocab=list(ipa_vocab)))


def ipa_metadata_audit(config, current_config):
    """Verify all phone bindings without acoustic payload/model/G2P execution."""
    from training import v2_fastpitch_ipa_experiment as ipa
    features=Path(current_config['args']['features'])
    rows={name:[json.loads(line) for line in (features/(name+'.jsonl')).read_text().splitlines() if line.strip()]
          for name in ('train','validation')}
    requested=current_config['args']
    bound=ipa.install_phone_context(requested['phone_manifest'],requested['phone_manifest_sha256'],rows,features)
    checks=dict(ipa_frontend_binding_exact=config.get('frontend')==bound,
                ipa_vocab_and_model_shape=(config['features']['vocab']==ipa.VOCAB
                    and config['model']['n_symbols']==len(ipa.VOCAB)==161),
                ipa_grapheme_verifier_source_unchanged=sha_file(ipa.grapheme_source.__file__)==ipa.GRAPHEME_SOURCE_SHA256,
                ipa_all_phone_rows_bound=True,ipa_all_original_glyph_ids_bound=True,
                ipa_all_phone_lengths_fit_MAS=True)
    count=0
    for split,feature_rows in rows.items():
        for feature in feature_rows:
            row=ipa._PHONE_ROWS[feature['id']]
            ids=ipa.validate_phone_ids(row['phoneme_ids'])
            matches=(row['split']==split and row['acoustic_feature_path']==feature['feature']
                     and row['acoustic_feature_sha256']==feature['feature_sha256']
                     and row['frames']==feature['frames']
                     and row['group_id']==feature.get('group_id')
                     and row['original_grapheme_text_length']==feature['text_length']
                     and row['map_sha256']==ipa.ipa_frontend.MAP_SHA256
                     and len(ids)==row['phone_length'])
            checks['ipa_all_phone_rows_bound']=checks['ipa_all_phone_rows_bound'] and matches
            glyph=ipa.grapheme_source.tokens(row['source_text']).tolist()
            checks['ipa_all_original_glyph_ids_bound']=(checks['ipa_all_original_glyph_ids_bound']
                and glyph==row['original_grapheme_ids'] and len(glyph)==feature['text_length']
                and row['glyph_normalized_text']==ipa.grapheme_source.normalize(row['source_text']))
            checks['ipa_all_phone_lengths_fit_MAS']=(checks['ipa_all_phone_lengths_fit_MAS']
                and 0<len(ids)<=feature['frames'])
            count+=1
    return checks,dict(frontend=bound,metadata_rows=count,
                      scope='Phone sidecar/input metadata, no acoustic payload read, no G2P or forward',
                      original_glyph_ids='Verified as acoustic source metadata; not model input',
                      actual_model_input='Frozen phoneme_ids; startup full acoustic+glyph audit precedes RAM-only override')


def audit_once(run,torch,np):
    path=run/'last.pt';receipt_path=run/'checkpoint.json';config_path=run/'config.json'
    receipt_bytes=receipt_path.read_bytes();receipt=json.loads(receipt_bytes)
    config_bytes=config_path.read_bytes();current_config=json.loads(config_bytes)
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno());opened_id=identity(opened)
        def stable():
            try:
                return (identity(os.fstat(stream.fileno()))==opened_id and identity(path.stat())==opened_id
                        and receipt_path.read_bytes()==receipt_bytes and config_path.read_bytes()==config_bytes)
            except FileNotFoundError:return False
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if not stable():raise SnapshotChanged('Checkpoint/receipt/config changed during full SHA read')
        digest=h.hexdigest()
        if digest!=receipt.get('sha256'):
            if opened.st_mtime_ns>receipt_path.stat().st_mtime_ns:
                raise SnapshotChanged('New checkpoint may be published before matching receipt')
            raise ValueError('Stable checkpoint SHA does not match receipt')
        stream.seek(0)
        # Locally produced trusted artifact; contains NumPy/Python RNG state.
        checkpoint=torch.load(stream,map_location='cpu',weights_only=False)
        if not stable():raise SnapshotChanged('Checkpoint/receipt/config changed during deserialization')
        required={'model','optimizer','scheduler','scaler','rng','step','epoch','cursor','best','bad','config','provenance','lr_state'}
        missing=sorted(required-set(checkpoint))
        checks={'required_state_present':not missing,'full_sha_matches_receipt':digest==receipt['sha256'],
                'step_matches_receipt':checkpoint.get('step')==receipt.get('step'),
                'epoch_matches_receipt':checkpoint.get('epoch')==receipt.get('epoch'),
                'cursor_matches_receipt':checkpoint.get('cursor')==receipt.get('cursor'),
                'cpu_only_requested':os.environ.get('CUDA_VISIBLE_DEVICES')=='',
                'cuda_not_initialized':not torch.cuda.is_initialized()}
        if missing:raise ValueError('Missing checkpoint state: '+repr(missing))
        config=checkpoint['config'];step=checkpoint['step']
        resource_changes,scientific_changes=compare_recipe(config,current_config)
        resource_authorization=None;resource_authorized=not resource_changes
        if resource_changes:
            authorization_path=run/f'resume-resource-change-{step}.json'
            if authorization_path.exists():
                resource_authorization=json.loads(authorization_path.read_text())
                declared=resource_authorization.get('changes',{})
                resource_authorized=(current_config.get('args',{}).get('resume_allow_resource_change') is True
                   and resource_authorization.get('checkpoint_sha256')==digest
                   and all(declared.get(k)==value for k,value in resource_changes.items())
                   and all(k in {'batch_size','accumulation','precision'} for k in declared)
                   and all(value==dict(before=config.get(k),after=current_config.get(k)) for k,value in declared.items())
                   and all(isinstance(current_config.get(k),int) and current_config[k]>0 for k in ('batch_size','accumulation'))
                   and current_config.get('precision') in ('bf16','fp16','fp32'))
        checks['config_matches_current_recipe']=not scientific_changes and resource_authorized
        checks['config_nonresource_fields_unchanged']=not scientific_changes
        checks['resource_changes_have_source_authorization']=resource_authorized
        checks['best_matches_receipt']=checkpoint['best']==receipt.get('best')
        checks['finite_best_and_valid_counters']=math.isfinite(checkpoint['best']) and all(isinstance(checkpoint[k],int) and checkpoint[k]>=0 for k in ('step','epoch','cursor','bad'))
        checks['scheduler_state_present']=isinstance(checkpoint['scheduler'],dict) and {'last_epoch','base_lrs','_last_lr'}<=set(checkpoint['scheduler'])
        checks['scheduler_step_matches']=checkpoint['scheduler'].get('last_epoch')==step
        checks['scaler_state_present']=isinstance(checkpoint['scaler'],dict)
        if config['precision']=='fp16':checks['fp16_scaler_complete']={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}<=set(checkpoint['scaler'])
        checks['lr_state_present']={'multiplier','reductions'}<=set(checkpoint['lr_state'])
        rng=checkpoint['rng'];checks['rng_complete']={'python','numpy','torch','cuda'}<=set(rng)
        rng_errors=[]
        try:
            random.Random().setstate(rng['python'])
            np.random.RandomState().set_state(rng['numpy'])
            torch.Generator(device='cpu').set_state(rng['torch'])
            assert isinstance(rng['cuda'],list) and len(rng['cuda'])>=1
            assert all(torch.is_tensor(v) and v.device.type=='cpu' and v.dtype==torch.uint8 and v.numel()>0 for v in rng['cuda'])
        except Exception as error:rng_errors.append(repr(error))
        checks['rng_state_valid_without_restore']=not rng_errors
        counts={};failures=[]
        def finite(value,label):
            count=0;elements=0;bytes_=0;nonlocal_failures=[]
            stack=[(label,value)]
            while stack:
                location,item=stack.pop()
                if torch.is_tensor(item):
                    count+=1;elements+=item.numel();bytes_+=item.numel()*item.element_size()
                    if item.device.type!='cpu':nonlocal_failures.append(location+':not_cpu')
                    elif (item.is_floating_point() or item.is_complex()) and not bool(torch.isfinite(item).all()):nonlocal_failures.append(location+':nonfinite')
                elif isinstance(item,dict):stack.extend((location+'.'+str(k),v) for k,v in item.items())
                elif isinstance(item,(tuple,list)):stack.extend((location+'.'+str(i),v) for i,v in enumerate(item))
                elif isinstance(item,float) and not math.isfinite(item):nonlocal_failures.append(location+':nonfinite_scalar')
            counts[label]=dict(tensors=count,elements=elements,tensor_bytes=bytes_)
            failures.extend(nonlocal_failures)
            return not nonlocal_failures
        checks['model_all_finite_cpu']=finite(checkpoint['model'],'model')
        checks['optimizer_all_finite_cpu']=finite(checkpoint['optimizer'],'optimizer')
        for key in ('scheduler','scaler','lr_state'):checks[key+'_all_finite_cpu']=finite(checkpoint[key],key)
        optimizer=checkpoint['optimizer'];groups=optimizer.get('param_groups',[])
        checks['optimizer_structure_valid']=bool(groups) and bool(optimizer.get('state'))
        optimizer_steps=sorted({float(state['step']) for state in optimizer.get('state',{}).values() if 'step' in state})
        checks['optimizer_steps_valid']=bool(optimizer_steps) and all(0<s<=step for s in optimizer_steps)
        rates=[float(g['lr']) for g in groups];scheduler_rates=checkpoint['scheduler'].get('_last_lr',[])
        expected_rate=config['lr']*min(1.,(checkpoint['scheduler']['last_epoch']+1)/config['warmup'])*checkpoint['lr_state']['multiplier']
        checks['learning_rates_consistent']=len(rates)==len(scheduler_rates) and all(math.isclose(rate,expected_rate,rel_tol=1e-10,abs_tol=1e-14) and math.isclose(rate,scheduler_rate,rel_tol=1e-10,abs_tol=1e-14) for rate,scheduler_rate in zip(rates,scheduler_rates))
        selection=config['selection'];reductions=checkpoint['lr_state']['reductions']
        checks['lr_reduction_state_consistent']=isinstance(reductions,int) and 0<=reductions<=selection['max_lr_reductions'] and math.isclose(checkpoint['lr_state']['multiplier'],selection['plateau_lr_factor']**reductions,rel_tol=1e-10)
        features=Path(current_config['args']['features']);manifest_checks={}
        for name,expected in config['features_manifest_sha256'].items():
            manifest=features/(name+'.jsonl');actual=sha_file(manifest)
            manifest_checks[str(manifest)]=dict(expected=expected,actual=actual,matches=actual==expected)
        for name,expected in config['features']['source_manifests'].items():
            actual=sha_file(name);manifest_checks[name]=dict(expected=expected,actual=actual,matches=actual==expected)
        current_features=json.loads((features/'features.json').read_text())
        from training import v2_fastpitch_ipa_experiment as ipa
        checks['feature_config_unchanged']=acoustic_config_matches(current_features,config,ipa.VOCAB)
        ipa_checks,ipa_binding=ipa_metadata_audit(config,current_config)
        checks.update(ipa_checks)
        checks['feature_and_source_manifest_hashes_match']=all(item['matches'] for item in manifest_checks.values())
        train_rows=[json.loads(line) for line in (features/'train.jsonl').read_text().splitlines() if line.strip()]
        checks['cursor_in_training_partition']=0<=checkpoint['cursor']<=len(train_rows)
        selection_receipt=config['features'].get('clean_selection',{})
        checks['clean_selection_receipt_matches']=bool(selection_receipt) and sha_file(selection_receipt['path'])==selection_receipt['sha256']
        source=Path(current_config['args']['source'])
        checks['upstream_source_manifest_hash_matches']=sha_file(source/'UPSTREAM.json')==checkpoint['provenance']['source_manifest_sha256']
        upstream=json.loads((source/'UPSTREAM.json').read_text())
        checks['upstream_source_pin_matches']=upstream['commit']==config['features']['upstream']==checkpoint['provenance']['upstream']
        checks['upstream_source_files_unchanged']=all(sha_file(source/item['path'])==item['sha256'] for item in upstream['files'])
        signature=[dict(name=name,shape=list(value.shape),dtype=str(value.dtype),elements=value.numel()) for name,value in sorted(checkpoint['model'].items())]
        signature_sha=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        model_config=config['model'];weights=checkpoint['model']
        checks['model_signature_matches_config']=(tuple(weights['encoder.word_emb.weight'].shape)==(model_config['n_symbols'],model_config['symbols_embedding_dim'])
             and tuple(weights['proj.weight'].shape)==(model_config['n_mel_channels'],model_config['out_fft_output_size']))
        checks['saved_code_provenance_present']=all(checkpoint['provenance'].get(k) for k in ('code_sha256','source_manifest_sha256','upstream','torch','cuda','argv','utc'))
        checks['ipa_saved_trainer_source_matches']=sha_file(ipa.__file__)==checkpoint['provenance'].get('code_sha256')
        # Instantiate only CPU to recover the exact parameter ordering/shape
        # expected by AdamW(model.parameters()). No forward or update occurs.
        from training.v2_fastpitch_ipa_experiment import imports
        import inspect
        cls,*_=imports(source);template=cls(**model_config)
        expected=template.state_dict()
        checks['model_state_schema_complete']=(set(weights)==set(expected) and all(
             tuple(weights[k].shape)==tuple(v.shape) and weights[k].dtype==v.dtype for k,v in expected.items()))
        # Pinned upstream declares this projection but never reads it in forward;
        # these two dead parameters legitimately never receive an Adam state.
        unused={'attention.attn_proj.weight','attention.attn_proj.bias'}
        checks['unused_parameter_exception_source_verified']=('attn_proj' not in inspect.getsource(type(template.attention).forward))
        layout_checks,optimizer_layout=adam_layout(optimizer,list(template.named_parameters()),weights,step,torch,unused)
        checks.update(layout_checks);del template
        checks['checkpoint_and_receipt_still_same_object']=stable()
        if not checks['checkpoint_and_receipt_still_same_object']:raise SnapshotChanged('Publication changed during tensor/config audit')
        checks['cuda_not_initialized_after']=not torch.cuda.is_initialized()
        return dict(status='pass' if all(checks.values()) else 'failed',checkpoint=str(path),sha256=digest,bytes=opened.st_size,
                    inode=opened.st_ino,step=checkpoint['step'],epoch=checkpoint['epoch'],cursor=checkpoint['cursor'],
                    checks=checks,tensor_counts=counts,model_signature_sha256=signature_sha,model_signature=signature,
                    optimizer_steps=optimizer_steps,learning_rates=rates,lr_state=checkpoint['lr_state'],
                    scheduler_last_epoch=checkpoint['scheduler']['last_epoch'],manifest_checks=manifest_checks,
                    config_resource_changes=resource_changes,config_scientific_changes=scientific_changes,
                    config_exact_match=not resource_changes and not scientific_changes,
                    resource_authorization=resource_authorization,
                    optimizer_layout=optimizer_layout,
                    tensor_failures=failures,rng_errors=rng_errors,checkpoint_provenance=checkpoint['provenance'],
                    ipa_binding=ipa_binding,
                    extra_scalar_state=dict(step=checkpoint['step'],epoch=checkpoint['epoch'],cursor=checkpoint['cursor'],
                        bad=checkpoint['bad'],best=checkpoint['best'],lr_state=checkpoint['lr_state'],
                        scheduler_last_epoch=checkpoint['scheduler']['last_epoch'],learning_rates=rates,
                        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        source_sha256=checkpoint['provenance']['code_sha256']),
                    note='Audit of saved state only; progress.json may contain later unsaved updates. No model forward, optimizer step, RNG restore on GPU or training-file mutation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    import torch,numpy as np
    torch.set_num_threads(1)
    attempts=[];result=None;exit_code=1;started=time.monotonic()
    for attempt in range(1,4):
        try:
            result=audit_once(args.run.resolve(),torch,np);exit_code=0 if result['status']=='pass' else 1
            attempts.append(dict(attempt=attempt,outcome=result['status']));break
        except SnapshotChanged as error:
            attempts.append(dict(attempt=attempt,outcome='transient',error=str(error)));exit_code=2
            result=dict(status='transient',checkpoint=str(args.run/'last.pt'),checks={},error=str(error))
            if attempt<3:time.sleep(1)
        except Exception as error:
            attempts.append(dict(attempt=attempt,outcome='failed',error=repr(error)));exit_code=1
            result=dict(status='failed',checkpoint=str(args.run/'last.pt'),checks={},error=repr(error));break
    for key in ('sha256','step','epoch','cursor','bytes'):result.setdefault(key,None)
    result.update(utc=dt.datetime.now(dt.timezone.utc).isoformat(),attempts=attempts,elapsed_seconds=time.monotonic()-started,
                  cuda_initialized=torch.cuda.is_initialized(),audit_code_sha256=sha_file(__file__),scope='CPU read-only complete saved FastPitch state audit using one stable descriptor')
    atomic_output(args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('model_signature','checkpoint_provenance','manifest_checks')},ensure_ascii=False),flush=True)
    raise SystemExit(exit_code)


if __name__=='__main__':main()
