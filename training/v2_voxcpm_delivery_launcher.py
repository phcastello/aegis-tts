"""Own one fresh Vox gateway, run the gated delivery client in this worker PID, clean up.

Default prepares validation metadata only. It never calls the historical probe's
hardcoded benchmark or changes production. Real execution needs the GPU worker.
"""
from __future__ import annotations
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from training import v2_delivery_protocol as protocol
from training import v2_http_delivery_benchmark as client
from training import v2_voxcpm_gateway_probe as gateway

BASE_SHA='f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d'
VAE_SHA='94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1'

def decorate(args):
    args.python=args.python.absolute()  # Preserve the venv symlink/sys.prefix.
    for name in ('run','checkpoint','source','model','cache','overlay','compiler_prerequisites'):
        path=getattr(args,name)
        if path is not None:setattr(args,name,path.resolve())
    if args.port!=8002:raise ValueError('Experimental fresh gateway restricted to port8002')
    if not 0<args.ready_timeout<=1800:raise ValueError('Readiness bound must be positive and <=1800s')
    args.endpoint='http://127.0.0.1:8002/v1/audio/speech';args.model_sha256=args.checkpoint_sha256
    front={'original-nfc':'OriginalNFC','f5-refinement-1':'F5Refinement1','ptbr-numbers-v1':'PtbrNumbers1'}[args.frontend]
    args.profile=f'AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-{front}-Experimental'
    args.startup_evidence=args.run/'startup-real.json';return args

def build_plan(args):
    settings=client.settings(args)
    settings['launcher']=dict(python=str(args.python),source=str(args.source),model=str(args.model),checkpoint=str(args.checkpoint),
                              overlay=str(args.overlay),cache=str(args.cache),compiler_prerequisites=str(args.compiler_prerequisites),
                              startup_ms=args.startup_ms,ready_timeout=args.ready_timeout,
                              lifecycle='Worker→launcher; launcher owns uvicorn child and directly calls client.execute in same PID; always stops owned server')
    sources=[Path(__file__),Path(client.__file__),Path(gateway.__file__),*[protocol.REPO/p for p in gateway.RUNTIME_FILES],
             protocol.REPO/'gateway/aegis_speech/normalization.py',protocol.REPO/'gateway/aegis_speech/segmentation.py']
    return protocol.make_plan(args,'voxcpm2-http',settings,sources)

def child_command(args):
    return [str(args.python),'-m','uvicorn','default_reference_gateway:app','--host','127.0.0.1','--port',str(args.port),
            '--workers','1','--loop','asyncio','--http','h11','--no-access-log','--log-level','warning']

def child_environment(args,config_path):
    env=gateway.child_environment(args)
    env.update(AEGIS_TTS_BACKEND='voxcpm2',AEGIS_VOXCPM_CONFIG=str(config_path),AEGIS_TTS_ONSET_TRIM='off',AEGIS_TTS_PROFILE_LATENCY='1',
               AEGIS_TTS_SOURCE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=protocol.REPO,text=True).strip())
    return env

def exact_child(current,expected):
    return all(current.get(k)==expected.get(k) for k in ('pid','ppid','start_ticks','uid','boot_id','command')) and current['uid']==os.getuid() and current['ppid']==os.getpid()

def stop_owned(process,expected):
    if process.poll() is not None:return process.returncode
    if not exact_child(protocol.identity(process.pid),expected):raise RuntimeError('Refuse TERM: actual child UID/start/argv/boot/parent differs')
    process.send_signal(signal.SIGTERM)
    try:return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if not exact_child(protocol.identity(process.pid),expected):raise RuntimeError('Refuse KILL: child identity changed after TERM')
        process.kill();return process.wait(timeout=10)

def artifact_bindings(args,config_path):
    files=[('checkpoint',args.checkpoint),('runtime_config',config_path),('base_model',args.model/'model.safetensors'),('audio_vae',args.model/'audiovae.pth')]
    files.extend(('model_metadata',args.model/name) for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','tokenization_voxcpm2.py'))
    bindings=[dict(role=role,path=str(path),sha256=protocol.sha(path)) for role,path in files]
    expected=dict(checkpoint=args.checkpoint_sha256,base_model=BASE_SHA,audio_vae=VAE_SHA)
    if any(b['role'] in expected and b['sha256']!=expected[b['role']] for b in bindings):raise ValueError('Selected/base/VAE checkpoint integrity failure')
    return bindings

def execute(args,plan):
    owner=protocol.owner_guard();protocol.gpu_set([]);preflight=protocol.budget(args.run,1024**3+32*1024**2)
    gateway.verify_overlay(args.overlay);gateway.cpu_check(args)
    sidecar=protocol.read(args.checkpoint.with_suffix('.sha256.json'))
    if sidecar.get('sha256')!=args.checkpoint_sha256:raise ValueError('Selected checkpoint sidecar differs')
    with socket.socket() as check:check.bind(('127.0.0.1',args.port))
    config=dict(source=str(args.source),model=str(args.model),adapter=str(args.checkpoint),adapter_sha256=args.checkpoint_sha256,
                cache=str(args.cache),gpu_lock=str(args.run.parent/'gateway-gpu.lock'),compiler_prerequisites=str(args.compiler_prerequisites),
                dsp='passthrough',frontend=args.frontend,instrument_replays=True,seed=protocol.SEED)
    config_path=args.run/'runtime.private.json';protocol.save(config_path,config);bindings=artifact_bindings(args,config_path)
    sources=protocol.source_bindings(protocol.REPO/p for p in gateway.RUNTIME_FILES)
    command=child_command(args);env=child_environment(args,config_path)
    process=None;identity=None;log_thread=None;watcher=None;sampler=None;stop=threading.Event();error=None;cleanup_error=None;client_returned=False
    logger=gateway.LimitedLog(args.run/'gateway.log')
    protocol.save(args.run/'launcher-recipe.json',dict(utc=protocol.utc(),owner=owner,preflight=preflight,plan_sha256=protocol.sha(args.run/'plan.json'),
                  command=command,runtime_config_sha256=protocol.sha(config_path),artifact_bindings=bindings,
                  runtime_source_hashes=sources,scope='Actual startup/readiness; common client in same immediate GPU-worker child PID; stop-only-owned uvicorn'))
    try:
        with protocol.metrics().GpuSampler(args.run/'startup-gpu.jsonl',True) as sampler:
            spawn_utc=protocol.utc();begun=time.perf_counter()
            with gateway.defer_spawn_signals():
                process=subprocess.Popen(command,cwd=protocol.REPO,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                identity=protocol.identity(process.pid)
            protocol.save(args.run/'owned-server.json',identity)
            log_thread=threading.Thread(target=logger.pump,args=(process.stdout,),daemon=True);log_thread.start()
            def watch():
                while not stop.wait(5):
                    try:protocol.output_guard(args.run)
                    except Exception as exc:
                        protocol.save(args.run/'launcher-safety-stop.json',dict(utc=protocol.utc(),reason=repr(exc)))
                        os.kill(os.getpid(),signal.SIGTERM);return
            watcher=threading.Thread(target=watch,daemon=True);watcher.start()
            polls=[];deadline=time.monotonic()+args.ready_timeout
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Owned gateway exited during startup')
                try:
                    status,headers,_=gateway.complete_request(args.port,'/ready',timeout=2)
                    polls.append(dict(elapsed_seconds=time.perf_counter()-begun,status=status))
                    if headers.get('x-aegis-tts-backend-state')=='failed':raise RuntimeError('Backend initialization failed; inspect bounded gateway.log')
                    if status==200:
                        protocol.save(args.run/'ready-launcher-response.json',dict(status=status,headers=headers))
                        client.readiness_contract(args,status,headers)
                        break
                except (OSError,http.client.HTTPException):pass
                protocol.save(args.run/'startup-progress.json',dict(phase='startup',elapsed_seconds=time.perf_counter()-begun,pid=process.pid));time.sleep(1)
            else:raise TimeoutError('Readiness timeout')
            ready_seconds=time.perf_counter()-begun;ready_utc=protocol.utc()
        if sampler.error:raise RuntimeError('Startup GPU sampler failed: '+sampler.error)
        protocol.gpu_set([process.pid]);protocol.save(args.run/'startup-gpu-audit.json',protocol.audit_gpu_samples(args.run/'startup-gpu.jsonl',[process.pid]))
        startup=dict(process=identity,boot_id=identity['boot_id'],spawn_utc=spawn_utc,ready_utc=ready_utc,spawn_to_ready_seconds=ready_seconds,
                     internal_warmup=dict(count=3,description='Actual pinned Vox backend boot executes three noncorpus calls on its persistent inference thread; separate from common external3'),
                     endpoint=args.endpoint,candidate=args.candidate,model_sha256=args.model_sha256,seed_fixed=protocol.SEED,
                     runtime_source_hashes=sources,artifact_bindings=bindings,polls=polls,sampled_startup_peak_mib=sampler.peak,
                     cold_scope='Fresh service process, existing OS/CUDA caches retained; no external synthesis before client warmups[0]')
        protocol.save(args.startup_evidence,startup)
        # Do not Popen the client: its immediate parent must remain the owner
        # GPU worker. This launcher and client share the same worker child PID.
        client.execute(args,plan);client_returned=True
        measured=protocol.read(args.run/'execution.json');expected=plan['cohort']['case_denominator']*args.repetitions
        if measured.get('error') or measured.get('observations')!=expected or measured.get('successes')!=expected:
            raise ValueError('Full delivery matrix contains failed/missing observations; preserved denominator, no pass claim')
        protocol.save(args.run/'post-matrix-contracts.json',gateway.contracts(args,args.profile))
        with (args.run/'smoke.log').open('wb') as log:
            result=subprocess.run([str(args.python),str(protocol.REPO/'scripts/smoke-test.py'),'--base-url',f'http://127.0.0.1:{args.port}'],cwd=protocol.REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError('Post-matrix actual smoke contracts failed')
    except BaseException as exc:error=repr(exc);raise
    finally:
        stop.set()
        if watcher:watcher.join(6)
        try:
            if process is not None and identity is not None:stop_owned(process,identity)
        except Exception as exc:cleanup_error=repr(exc)
        if log_thread:log_thread.join(5)
        same=protocol.source_bindings(plan['source_hashes'])==plan['source_hashes']
        checkpoint_unchanged=protocol.sha(args.checkpoint)==args.checkpoint_sha256
        still_running=process is not None and process.poll() is None
        gpu_empty=False;release_deadline=time.monotonic()+10
        while time.monotonic()<release_deadline:
            try:gpu_empty=not protocol.metrics().gpu_snapshot()['processes']
            except Exception:break
            if gpu_empty:break
            time.sleep(.2)
        final_budget_error=None
        try:final_budget=protocol.budget(args.run,0)
        except Exception as exc:final_budget_error=repr(exc);final_budget=dict(error=final_budget_error)
        protocol.save(args.run/'launcher-result.json',dict(utc=protocol.utc(),error=error or final_budget_error,cleanup_error=cleanup_error,client_returned=client_returned,
                      checkpoint_unchanged=checkpoint_unchanged,sources_unchanged=same,server_left_running=still_running,gpu_empty_after_cleanup=gpu_empty,
                      owned_server=identity,server_returncode=process.poll() if process is not None else None,
                      log_total_bytes=logger.total_bytes,log_truncations=logger.truncations,
                      final_working=final_budget,final_budget_error=final_budget_error,
                      scope='Complete only if matrix execution has no error, post-matrix contracts/smoke pass and cleanup verifies GPU empty'))
        if cleanup_error or not same or not checkpoint_unchanged or still_running or not gpu_empty or final_budget_error:raise RuntimeError('Owned lifecycle/source/checkpoint/GPU/budget cleanup failed; inspect launcher-result.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__);protocol.common_arguments(parser)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--python',type=Path,default=gateway.PYTHON);parser.add_argument('--overlay',type=Path,default=protocol.ROOT/'voxcpm-gateway/deps')
    parser.add_argument('--source',type=Path,default=protocol.ROOT/'voxcpm2/source');parser.add_argument('--model',type=Path,default=protocol.ROOT/'voxcpm2/model')
    parser.add_argument('--cache',type=Path,default=protocol.ROOT/'voxcpm2/compile-cache');parser.add_argument('--compiler-prerequisites',type=Path,default=protocol.ROOT/'voxcpm2/compile-prereqs.json')
    parser.add_argument('--frontend',choices=('original-nfc','f5-refinement-1','ptbr-numbers-v1'),default='ptbr-numbers-v1')
    parser.add_argument('--startup-ms',type=gateway.startup_ms,default=320.);parser.add_argument('--port',type=int,default=8002)
    parser.add_argument('--ready-timeout',type=float,default=600.);parser.add_argument('--timeout',type=float,default=180.)
    args=decorate(parser.parse_args());os.umask(0o077);plan=build_plan(args)
    if protocol.plan_or_verify(args,plan):
        with gateway.cleanup_signals():execute(args,plan)
    print(json.dumps(dict(status='executed' if args.action=='execute' else 'prepared_only',run=str(args.run),gpu_executed=args.action=='execute')))

if __name__=='__main__':main()
