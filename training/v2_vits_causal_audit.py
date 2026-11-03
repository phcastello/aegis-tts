"""CPU-only VITS batch, losses, gradient and prior drift diagnostics.

Does not change training recipes/manifests/checkpoints or use CUDA.
"""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
from types import SimpleNamespace
from training import v2_vits as v


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('/var/tmp/aegis-v2-work/portuguese/vits-prepared'))
    p.add_argument('--output',type=Path,default=v.ROOT/'causal-audit/result.json');a=p.parse_args()
    os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']=''
    args=SimpleNamespace(root=v.ROOT,device='cpu',gpu_authorized=False,threads=2)
    torch,np,m,c,n=v.bootstrap(args)
    base={k:t.detach().clone() for k,t in m.state_dict().items()}
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    trained={k:t for k,t in ckpt['model'].items() if not k.startswith('disc.')}
    result=dict(checkpoint=str(a.checkpoint),checkpoint_step=ckpt['step'],checkpoint_epoch=ckpt['epoch'],
                checkpoint_sha256=v.digest(a.checkpoint),cpu_only=True,no_training_artifacts_modified=True,
                source_code_sha256=v.digest(v.__file__))
    del ckpt
    rows,_=v.rows_and_fingerprint(a.prepared,m,n);row=rows['train'][0]
    # Collate/format the exact same waveform and effective tokens using the
    # official VITS dataset path, independently of the local batch constructor.
    from TTS.tts.models.vits import VitsDataset
    audio=v.get_audio(row,np)
    item=dict(raw_text=row['text'],token_ids=row['_tokens'],token_len=len(row['_tokens']),
              wav=torch.from_numpy(audio)[None],wav_file=Path(row['audio_filepath']).name,
              speaker_name='male',language_name='pt',audio_unique_name=row['id'])
    upstream=VitsDataset.collate_fn(SimpleNamespace(pad_id=m.tokenizer.characters.pad_id),[item])
    upstream=m.format_batch_on_device(m.format_batch(upstream))
    own=v.batch(row,m,torch,np,'cpu')
    keys=('tokens','token_lens','waveform','speaker_ids','language_ids','spec','mel','spec_lens','mel_lens')
    result['batch_comparison']=dict(sample_id=row['id'],checks={key:torch.equal(upstream[key],own[key]) for key in keys},
                                   shapes={key:list(own[key].shape) for key in keys},
                                   peak=float(abs(audio).max()),rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                                   waveform_normalization='PCM scaling and resample only, no gain/trim, matching official VitsDataset.load_audio semantics')
    groups={}
    for key,b in base.items():
        group=key.split('.')[0];item=groups.setdefault(group,dict(base_squared=0.,delta_squared=0.,max_abs_delta=0.,elements=0))
        diff=trained[key].double()-b.double();item['base_squared']+=float(b.double().square().sum())
        item['delta_squared']+=float(diff.square().sum());item['max_abs_delta']=max(item['max_abs_delta'],float(abs(diff).max()))
        item['elements']+=b.numel()
    result['weight_drift']={k:{**x,'relative_l2':math.sqrt(x['delta_squared']/max(1e-30,x['base_squared']))} for k,x in groups.items()}
    cases=json.loads((v.REPO/'benchmarks/v2/validation.pt-BR.json').read_text());priors={}
    for label,weights in [('base',base),('ptbr_epoch1',trained)]:
        m.load_state_dict(weights,strict=True);m.eval();values=[]
        for case in cases:
            torch.manual_seed(40733133)
            token=torch.tensor(v.tokens(m,n,case['text']))[None]
            with torch.no_grad():
                g=m.emb_g(torch.tensor([0])).unsqueeze(-1);lang=m.emb_l(torch.tensor([0])).unsqueeze(-1)
                encoded,prior,logs,mask=m.text_encoder(token,torch.tensor([token.shape[1]]),lang_emb=lang)
                logw=m.duration_predictor(encoded,mask,g=g,reverse=True,noise_scale=0.,lang_emb=lang)
                durations=torch.ceil(torch.exp(logw)*mask*m.length_scale)
            values.append(dict(case=case['id'],tokens=token.numel(),predicted_frames=int(durations.sum()),
                               predicted_seconds=float(durations.sum())*256/22050,max_token_frames=float(durations.max()),
                               duration_finite=bool(torch.isfinite(durations).all()),prior_rms=float(prior.square().mean().sqrt()),
                               log_sigma_mean=float(logs.mean()),log_sigma_std=float(logs.std())))
        priors[label]=values
    result['zero_noise_text_priors']=priors
    a.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    v.atomic_json(a.output.with_name('partial.json'),result)
    # A real official loss/backward on the trained model, with a fresh critic,
    # verifies target orientation and gradient paths; no weight update is saved.
    from TTS.tts.layers.vits.discriminator import VitsDiscriminator
    torch.manual_seed(20261103);m.disc=VitsDiscriminator(periods=c.model_args.periods_multi_period_discriminator,
                                                        use_spectral_norm=c.model_args.use_spectral_norm_disriminator)
    m.train();criteria=m.get_criterion();b=v.batch(row,m,torch,np,'cpu');started=time.perf_counter()
    _,dl=m.train_step(b,criteria[0],optimizer_idx=0)
    dl['loss'].backward();disc_grad_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.disc.parameters())
    for p in m.disc.parameters():p.requires_grad_(False)
    _,gl=m.train_step(b,criteria[1],optimizer_idx=1);gl['loss'].backward()
    grad={}
    for name,param in m.named_parameters():
        if name.startswith('disc.') or param.grad is None:continue
        group=name.split('.')[0];item=grad.setdefault(group,dict(norm_squared=0.,finite=True,elements=0))
        item['norm_squared']+=float(param.grad.double().square().sum());item['finite']&=bool(torch.isfinite(param.grad).all());item['elements']+=param.numel()
    scalar=lambda x:float(x.detach()) if torch.is_tensor(x) else float(x)
    result['loss_backward']=dict(generator_losses={k:scalar(x) for k,x in gl.items()},
                                 discriminator_losses={k:scalar(x) for k,x in dl.items()},disc_grad_finite=disc_grad_finite,
                                 gradient_groups={k:{**x,'l2_norm':math.sqrt(x['norm_squared'])} for k,x in grad.items()},
                                 seconds=time.perf_counter()-started,scope='official train_step/criterion/backward on one real public sample, fresh common critic; no optimizer update saved')
    a.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True);v.atomic_json(a.output,result)
    print(json.dumps(dict(output=str(a.output),batch=result['batch_comparison']['checks'],backward_seconds=result['loss_backward']['seconds'],
                          generator_losses=result['loss_backward']['generator_losses'],disc_grad_finite=disc_grad_finite,
                          gradient_finite=all(x['finite'] for x in grad.values()))),flush=True)


if __name__=='__main__':main()
