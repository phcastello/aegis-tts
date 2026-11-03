"""Explicit static-KV-capacity ablation around the unchanged VoxCPM2 file probe.

Default action only writes metadata. GPU execution remains exclusively inside
the existing owner queue, budget/idle guards, full-GPU loader and official
compile path. This wrapper changes one in-memory config field before that
loader recreates both KV caches. Source, weights, RoPE, GQA, mask and SDPA route
are unchanged. A speed benefit is a hypothesis until a local measurement.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

from training import v2_voxcpm_optimized_probe as optimized
from training import v2_voxcpm_probe as pilot
from training import v2_voxcpm_streaming_probe as native


def capacity_guard(text_tokens, max_new_patches, capacity, extra_tokens=2):
    """Zero-shot text + audio-start + full requested generation + one spare.

    The native ratio-based generation limit may be smaller. We intentionally
    reserve the full requested max_len and reject overflow instead of silently
    shortening output. Prompt/reference modes need their own prefix accounting.
    """
    values=(text_tokens,max_new_patches,capacity,extra_tokens)
    if any(type(value) is not int or value<0 for value in values):
        raise ValueError('Cache guard counts must be nonnegative integers')
    if not capacity or not max_new_patches or extra_tokens<2:
        raise ValueError('Positive capacity/max_len and at least2 extra slots required')
    needed=text_tokens+max_new_patches+extra_tokens
    if needed>capacity:
        raise ValueError(f'KV cache needs {needed} slots, capacity={capacity}; refuse truncation')
    return dict(text_tokens=text_tokens,max_new_patches=max_new_patches,
                extra_tokens=extra_tokens,reserved_slots=needed,capacity=capacity,
                spare_slots=capacity-needed,mode='zero-shot; no audio/text reference prefix')


def cache_record(model,capacity):
    caches={}
    for name in ('base_lm','residual_lm'):
        cache=getattr(model,name).kv_cache
        tensor=cache.kv_cache
        if cache.max_length!=capacity or tensor.shape[-2]!=capacity:
            raise RuntimeError('Requested KV capacity not reflected in '+name)
        caches[name]=dict(capacity=cache.max_length,shape=list(tensor.shape),
                          dtype=str(tensor.dtype),device=str(tensor.device),
                          bytes=tensor.numel()*tensor.element_size())
    return dict(config_max_length=model.config.max_length,caches=caches,
                kv_bytes=sum(value['bytes'] for value in caches.values()))


@contextlib.contextmanager
def capacity_hooks(capacity,extra_tokens,state):
    """Monkeyhooks exist only in this new process and are always restored."""
    original_move=pilot.model_to_gpu
    original_synthesize=pilot.synthesize
    original_residency=optimized.residency
    original_collect=native.collect_stream
    def move(model,torch):
        before=int(model.config.max_length)
        if capacity>before:raise ValueError('This ablation may only reduce native KV capacity')
        for name in ('base_lm','residual_lm'):
            step=getattr(model,name).forward_step
            if hasattr(step,'_torchdynamo_orig_callable') or hasattr(step,'_orig_mod'):
                raise RuntimeError('Change cache capacity before tracing/compilation')
        # max_length is used only for the two setup_cache calls in this pin;
        # lm_config.max_position_embeddings and RoPE buffers remain untouched.
        position_limit=model.config.lm_config.max_position_embeddings
        model.config.max_length=capacity
        model=original_move(model,torch)
        if model.config.lm_config.max_position_embeddings!=position_limit:
            raise RuntimeError('Cache move unexpectedly changed position/RoPE configuration')
        state.update(native_capacity=before,requested_capacity=capacity,
                     preserved_max_position_embeddings=position_limit,
                     actual=cache_record(model,capacity))
        generate=model._generate
        def guarded_generate(*args,**kwargs):
            if len(args)>1:raise ValueError('Use named zero-shot generation arguments')
            if any(kwargs.get(key) for key in ('prompt_text','prompt_wav_path','reference_wav_path')):
                raise ValueError('Cache ablation requires empty prompt/reference prefixes')
            text=kwargs.get('target_text',args[0] if args else None)
            if not isinstance(text,str):raise ValueError('Target text required for capacity guard')
            maximum=kwargs.get('max_len',2000)
            # Called by the original synthesize AFTER its timer starts.
            state['last_generation_guard']=capacity_guard(len(model.text_tokenizer(text)),maximum,capacity,extra_tokens)
            return generate(*args,**kwargs)
        # Both public generate and generate_streaming call this central entry.
        # The native PCM collector starts its clock before requesting it.
        model._generate=guarded_generate
        return model
    def residency(model,torch):
        record=original_residency(model,torch)
        record['static_kv_capacity_ablation']=cache_record(model,capacity)
        return record
    def synthesize(args,model,torch,np,label):
        row=original_synthesize(args,model,torch,np,label)
        row['static_kv_capacity_ablation']=dict(capacity=capacity,
            original_capacity=state['native_capacity'],guard=state['last_generation_guard'],
            guard_timing='inside existing synthesis timer; before native prefill')
        # The shared helper writes this same label's receipt before returning.
        pilot.atomic_json(args.run/(label+'.json'),row)
        return row
    def collect(*args,**kwargs):
        pcm,row=original_collect(*args,**kwargs)
        row['static_kv_capacity_ablation']=dict(capacity=capacity,
            original_capacity=state['native_capacity'],guard=state.get('last_generation_guard'),
            guard_timing='inside native PCM client timer, before native prefill')
        return pcm,row
    pilot.model_to_gpu=move
    pilot.synthesize=synthesize
    optimized.residency=residency
    native.collect_stream=collect
    try:yield
    finally:
        pilot.model_to_gpu=original_move
        pilot.synthesize=original_synthesize
        optimized.residency=original_residency
        native.collect_stream=original_collect


def cpu_attention_proof():
    """Tiny CPU GQA/mask equivalence, never load model weights or touch CUDA."""
    if (optimized.ROOT/'pause-asr').exists():
        raise RuntimeError('Defer CPU tensor tests until the GPU timing pause is over')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    generator=torch.Generator(device='cpu').manual_seed(7311026)
    q=torch.randn(1,4,1,8,dtype=torch.float64,generator=generator)
    keys=torch.randn(1,1,16,8,dtype=torch.float64,generator=generator)
    values=torch.randn(1,1,16,8,dtype=torch.float64,generator=generator)
    rows=[]
    for position in (0,2,7,11,15):
        outputs=[]
        for capacity in (16,128):
            k=torch.full((1,1,capacity,8),123.,dtype=torch.float64)
            v=torch.full((1,1,capacity,8),-57.,dtype=torch.float64)
            k[:,:,:position+1]=keys[:,:,:position+1]
            v[:,:,:position+1]=values[:,:,:position+1]
            mask=(torch.arange(capacity)<=position).view(1,1,1,-1)
            outputs.append(torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True))
        difference=float((outputs[0]-outputs[1]).abs().max())
        if not torch.allclose(outputs[0],outputs[1],rtol=1e-12,atol=1e-12):
            raise AssertionError('Masked future cache changed valid-prefix attention')
        rows.append(dict(position=position,max_abs_difference=difference))
    if torch.cuda.is_initialized():raise AssertionError('CPU proof unexpectedly initialized CUDA')
    return dict(status='pass',rows=rows,cuda_initialized=False,
        scope='Tiny float64 CPU GQA with identical valid prefix and masked arbitrary future values. Does not prove GPU bit identity, speed, pronunciation or voice quality.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','cpu-check','execute'],default='plan')
    p.add_argument('--source',type=Path,default=pilot.ROOT/'source')
    p.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    p.add_argument('--run',type=Path,default=pilot.ROOT/'cache1024-preparation')
    p.add_argument('--cache',type=Path,default=pilot.ROOT/'compile-cache')
    p.add_argument('--delivery',choices=['native','file'],default='native')
    p.add_argument('--kv-cache-capacity',type=int,default=1024)
    p.add_argument('--kv-extra-tokens',type=int,default=2)
    p.add_argument('--topology',choices=['base'],default='base')
    p.add_argument('--optimize',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--cache-limit-mib',type=int,default=512)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--nfe',type=int,choices=[10,6,4],default=4)
    p.add_argument('--matmul-precision',choices=['highest'],default='highest')
    p.add_argument('--cases',nargs='+',default=['v_reply','v_long'])
    p.add_argument('--cancel-after-native-chunks',type=int,default=0)
    a=p.parse_args();os.umask(0o077)
    if a.kv_cache_capacity<402 or a.kv_cache_capacity>8192 or a.kv_extra_tokens<2:
        p.error('Capacity must be402..8192 and reserve at least2 extra slots')
    if a.checkpoint:p.error('Base cache ablation must not silently load an adapter')
    if a.cancel_after_native_chunks<0:p.error('Cancel chunk count must be nonnegative')
    optimized.build_trials(a.cases,a.repetitions,a.optimize)
    a.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    state={}
    recipe=dict(utc=optimized.utc(),code_sha256=pilot.sha(__file__),
        unchanged_optimized_probe_sha256=pilot.sha(optimized.__file__),
        unchanged_native_probe_sha256=pilot.sha(native.__file__),
        unchanged_synthesis_helper_sha256=pilot.sha(pilot.__file__),
        source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,
        requested_capacity=a.kv_cache_capacity,extra_tokens=a.kv_extra_tokens,
        nfe=a.nfe,validation_cases=a.cases,repetitions=a.repetitions,delivery=a.delivery,
        hypothesis='A shorter static cache may reduce full-cache masked GQA/SDPA work. Inference from code until local profile/measurement.',
        changed='Only model.config.max_length before existing full-GPU loader recreates both caches and before optimize/first trace',
        preserved=['weights','RoPE and position buffers','GQA heads','SDPA implementation and mask','max_len400','NFE and CFG','precision','resample and PCM metrics'],
        guard='Central _generate: exact tokenizer count + requested max_len +2 <=capacity; reject, never truncate; empty prompt/reference prefixes only',
        warmup='Three excluded calls per case before warm matrix, inherited unchanged',
        quality='New inference configuration; CPU mathematical equivalence does not approve GPU audio/ASR/identity',
        gpu_executed=False)
    if a.action=='plan':
        pilot.atomic_json(a.run/'cache-plan.json',recipe)
        print(json.dumps(recipe,indent=2));return
    if a.action=='cpu-check':
        pilot.atomic_json(a.run/'cache-cpu-proof.json',dict(**recipe,proof=cpu_attention_proof()))
        return
    if (a.run/'cache-ablation.json').exists():raise ValueError('Use a fresh cache ablation directory')
    try:
        with capacity_hooks(a.kv_cache_capacity,a.kv_extra_tokens,state):
            (native.execute if a.delivery=='native' else optimized.execute)(a)
        recipe.update(status='complete',gpu_executed=True,actual_state=state)
    except BaseException as error:
        recipe.update(status='failed',gpu_executed=bool(state.get('actual')),
                      error=repr(error),traceback=traceback.format_exc(),actual_state=state)
        raise
    finally:pilot.atomic_json(a.run/'cache-ablation.json',recipe)


if __name__=='__main__':main()
