"""Thin native-streaming LoRA benchmark; default is a CPU metadata plan.

Uses the unchanged native runner, original LoRA branches (no merge), 1024 KV
slots and existing worker/idle/budget/compile guards. No weights are downloaded.
"""
import argparse
from pathlib import Path
import json
import os
from training import v2_voxcpm_probe as pilot
from training import v2_voxcpm_cache_probe as capacity
from training import v2_voxcpm_streaming_probe as native
from training import v2_voxcpm_optimized_probe as optimized


def checkpoint_receipt(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Checkpoint must be a real local file')
    receipt=json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))
    digest=pilot.sha(path)
    if receipt['sha256']!=digest or receipt.get('bytes',path.stat().st_size)!=path.stat().st_size:
        raise ValueError('Checkpoint/sidecar integrity mismatch')
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size,step=receipt.get('step'),sidecar=str(path.with_suffix('.sha256.json')))


def lora_residency(model,torch):
    parameters=[p for name,p in model.named_parameters() if 'lora_' in name]
    branches=sum(module.__class__.__name__=='LoRALinear' for module in model.modules())
    if not parameters or not branches:raise ValueError('Actual unmerged LoRA branches required')
    if any(p.dtype!=torch.bfloat16 or p.device.type!='cuda' or p.requires_grad for p in parameters):
        raise ValueError('Inference LoRA must be frozen BF16 entirely on CUDA')
    return dict(branches=branches,parameter_tensors=len(parameters),elements=sum(p.numel() for p in parameters),
                precision='BF16 inference copy; FP32 source checkpoint preserved',device='cuda',merged=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','execute'],default='plan');p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,default=pilot.ROOT/'source');p.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    p.add_argument('--run',type=Path,default=pilot.ROOT/'lora-native-preparation');p.add_argument('--cache',type=Path,default=pilot.ROOT/'compile-cache')
    p.add_argument('--nfe',type=int,choices=[4,6,10],default=10);p.add_argument('--rank',type=int,default=8)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--cases',nargs='+',default=['v_reply','v_long'])
    p.add_argument('--cache-limit-mib',type=int,default=512);p.add_argument('--gpu-authorized',action='store_true')
    a=p.parse_args();os.umask(0o077);a.run=a.run.resolve();a.checkpoint=a.checkpoint.resolve()
    a.topology='lora';a.optimize=True;a.matmul_precision='highest';a.cancel_after_native_chunks=0
    optimized.build_trials(a.cases,a.repetitions,True)
    if a.rank<1 or not 1<=a.cache_limit_mib<=512:p.error('Positive rank and compile cache cap1..512MiB required')
    if any((path/'.git').exists() for path in (a.run,*a.run.parents)):p.error('Private benchmark output must stay outside Git')
    a.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    before=checkpoint_receipt(a.checkpoint) if a.checkpoint.exists() else None
    recipe=dict(code_sha256=pilot.sha(__file__),native_code_sha256=pilot.sha(native.__file__),
        capacity_code_sha256=pilot.sha(capacity.__file__),optimized_code_sha256=pilot.sha(optimized.__file__),
        source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,checkpoint=before,
        pending_checkpoint=str(a.checkpoint) if before is None else None,topology='lora',merge=False,
        rank=a.rank,nfe=a.nfe,cfg=2.,seed=40733133,matmul_precision='highest',kv_capacity=1024,
        cache_limit_mib=a.cache_limit_mib,warmup='3 excluded calls per case; then warm matrix',
        delivery='Native streaming ->24k PCM client; all neural weights/VAE/KV on GPU; no HTTP',
        precision='Source LoRA FP32 strictly copied, inference branches BF16, frozen base BF16/VAEFP32',
        gpu_executed=False,quality='Engineering/validation evidence; ASR, content and identity remain required')
    if a.action=='plan':
        pilot.atomic_json(a.run/'wrapper-plan.json',recipe);print(json.dumps(recipe,indent=2));return
    if before is None:raise ValueError('Selected checkpoint does not exist yet')
    if (a.run/'wrapper-recipe.json').exists():raise ValueError('Fresh benchmark directory required')
    state={}
    try:
        with capacity.capacity_hooks(1024,2,state):
            original=optimized.residency
            def checked(model,torch):
                record=original(model,torch);state['lora_actual']=lora_residency(model,torch)
                state['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda)
                return dict(**record,lora_inference=state['lora_actual'])
            optimized.residency=checked
            try:native.execute(a)
            finally:optimized.residency=original
        recipe.update(status='complete',gpu_executed=True,actual=state)
    except BaseException as error:
        recipe.update(status='failed',gpu_executed=bool(state.get('actual')),actual=state,error=repr(error));raise
    finally:
        try:after=checkpoint_receipt(a.checkpoint)
        except Exception as error:
            recipe.update(source_checkpoint_unchanged=False,checkpoint_after_error=repr(error))
            pilot.atomic_json(a.run/'wrapper-recipe.json',recipe);raise
        recipe['checkpoint_after']=after
        recipe['source_checkpoint_unchanged']=after==before
        pilot.atomic_json(a.run/'wrapper-recipe.json',recipe)
        if after!=before:raise RuntimeError('Source checkpoint changed during inference; comparison invalid')


if __name__=='__main__':main()
