"""Client-only common delivery benchmark of an independently started fresh Vox gateway.

The caller owns startup, lifecycle and final availability. This executor never
starts/stops a server. Default plan is stdlib-only and validation7-only.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time
from training import v2_delivery_protocol as protocol
from training import v2_voxcpm_gateway_probe as gateway

def settings(args):
    protocol.metrics().endpoint(args.endpoint)
    if args.timeout<=0 or not math.isfinite(args.timeout):raise ValueError('Positive finite timeout required')
    if len(args.model_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.model_sha256):raise ValueError('Pinned model SHA256 required')
    return dict(endpoint=args.endpoint,voice_profile=args.profile,frontend=args.frontend,model_sha256=args.model_sha256,
                startup_evidence=str(args.startup_evidence.absolute()),timeout=args.timeout,
                backend='voxcpm2',internal_warmups='Startup receipt must record actual backend boot warmups separately',
                seed=protocol.SEED,payload=dict(stream=True,response_format='pcm'),
                scope='Client HTTP PCM; common receive/onset/FIFO rules; no hardware playback delay; no server lifecycle')

def startup_contract(args,value,current,boot):
    """Validate real PID/start/argv plus startup clocks, not merely a cold flag."""
    required={'process','boot_id','spawn_utc','ready_utc','spawn_to_ready_seconds','internal_warmup',
              'endpoint','candidate','model_sha256','seed_fixed','runtime_source_hashes','artifact_bindings'}
    if not isinstance(value,dict) or not required<=value.keys():raise ValueError('Incomplete fresh startup receipt')
    claimed=value['process']
    for key in ('pid','start_ticks','uid','command'):
        if str(claimed.get(key))!=str(current.get(key)):raise ValueError('Startup PID/start/UID/argv differs from live process')
    if value['boot_id']!=boot or value['boot_id']!=current['boot_id']:raise ValueError('Startup boot differs')
    if value['endpoint']!=args.endpoint or value['candidate']!=args.candidate or value['model_sha256']!=args.model_sha256:
        raise ValueError('Startup endpoint/candidate/checkpoint differs')
    if value['seed_fixed']!=protocol.SEED:raise ValueError('Actual fixed seed contract differs')
    start=dt.datetime.fromisoformat(value['spawn_utc']);ready=dt.datetime.fromisoformat(value['ready_utc'])
    seconds=value['spawn_to_ready_seconds']
    if start.tzinfo is None or ready.tzinfo is None or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0 or ready<start:
        raise ValueError('Invalid measured spawn/readiness clocks')
    if abs((ready-start).total_seconds()-seconds)>5:raise ValueError('Startup UTC/monotonic durations inconsistent')
    process_started=dt.datetime.fromisoformat(current['start_utc_estimated'])
    if abs((process_started-start).total_seconds())>10:raise ValueError('Spawn clock does not match actual process start; cold flag is insufficient')
    if not isinstance(value['internal_warmup'],dict) or not value['internal_warmup'].get('description') or value['internal_warmup'].get('count')!=3:
        raise ValueError('Actual mandatory internal warmup description required')
    if not isinstance(value['runtime_source_hashes'],dict) or not value['runtime_source_hashes']:
        raise ValueError('Runtime sources must be bound at actual startup')
    bindings=value['artifact_bindings']
    if not isinstance(bindings,list) or not any(b.get('role')=='checkpoint' and b.get('sha256')==args.model_sha256 for b in bindings):
        raise ValueError('Startup must bind the selected checkpoint artifact')
    return value

def live_config_path(pid):
    # Inspect only the named non-secret config variable; never expose environ.
    values=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    paths=[v.split(b'=',1)[1].decode() for v in values if v.startswith(b'AEGIS_VOXCPM_CONFIG=')]
    if len(paths)!=1:raise ValueError('Live server runtime config variable missing/duplicate')
    return Path(paths[0]).resolve()

def verified_startup(args):
    value=protocol.read(args.startup_evidence);claimed=value.get('process',{})
    current=protocol.identity(claimed.get('pid',-1));startup_contract(args,value,current,current['boot_id'])
    expected={str(protocol.REPO/p) for p in gateway.RUNTIME_FILES}
    if not expected<=set(value['runtime_source_hashes']):raise ValueError('Required gateway/runtime source hashes missing')
    for path,digest in value['runtime_source_hashes'].items():
        if protocol.sha(path)!=digest:raise ValueError('Runtime source differs from startup')
    for binding in value['artifact_bindings']:
        if protocol.sha(binding['path'])!=binding['sha256']:raise ValueError('Startup model/config artifact changed')
    configs=[b for b in value['artifact_bindings'] if b.get('role')=='runtime_config']
    checkpoints=[b for b in value['artifact_bindings'] if b.get('role')=='checkpoint' and b.get('sha256')==args.model_sha256]
    if len(configs)!=1 or len(checkpoints)!=1:raise ValueError('One actual runtime config and selected checkpoint required')
    config=protocol.read(configs[0]['path'])
    if config.get('adapter_sha256')!=args.model_sha256 or Path(config.get('adapter','')).resolve()!=Path(checkpoints[0]['path']).resolve() or config.get('frontend')!=args.frontend or config.get('seed',protocol.SEED)!=protocol.SEED:
        raise ValueError('Runtime config does not bind expected checkpoint/frontend')
    if live_config_path(current['pid'])!=Path(configs[0]['path']).resolve():raise ValueError('Live server uses a different runtime config')
    return value

def request_count(port):
    status,_,body=gateway.complete_request(port,'/metrics')
    if status!=200:raise ValueError('Request counters unavailable')
    values={line.split()[0]:float(line.split()[1]) for line in body.decode().splitlines() if line and not line.startswith('#')}
    count=values.get('aegis_tts_requests_total')
    if count is None or not math.isfinite(count) or count<0 or count!=int(count):raise ValueError('Invalid request counter')
    return int(count)

def completed_contract(args,row):
    headers=row.get('headers',{})
    if headers.get('x-aegis-voice-profile')!=args.profile or headers.get('x-aegis-tts-frontend')!=args.frontend:
        raise ValueError('Delivered voice profile/frontend differs')
    identifier=headers.get('x-aegis-request-id')
    if not identifier:raise ValueError('No request ID')
    port=protocol.metrics().endpoint(args.endpoint).port
    status,_,body=gateway.complete_request(port,'/v1/aegis/speech/'+identifier)
    receipt=gateway.status_receipt(row,status,body)
    if receipt.get('status')!='completed':raise ValueError('PCM HTTP200 did not complete at gateway')
    status,_,body=gateway.complete_request(port,'/v1/aegis/speech/'+identifier+'/events')
    if len(body)>65536:raise ValueError('Unexpected SSE evidence exceeds64KiB bound')
    events=body.decode('utf-8')
    named=[line.removeprefix('event:').strip() for line in events.splitlines() if line.startswith('event:')]
    if status!=200 or 'accepted' not in named or 'completed' not in named or any(e in named for e in ('failed','cancelled')):
        raise ValueError('SSE lacks accepted/completed or contains failure')
    return dict(status=receipt,sse=events)

def enforce_deadline(row,timeout):
    # The frozen receive() checks deadlines after nonempty chunks; EOF can
    # arrive later. Keep its timing/PCM evidence but reject the whole outcome.
    if row.get('outcome')=='ok' and row['eof_ms']>timeout*1000:
        row.update(outcome='failed_deadline',error='EOF exceeded overall client deadline; measured timing/PCM preserved as failure')
    return row

def execute(args,plan):
    owner=protocol.owner_guard();startup=verified_startup(args);pid=startup['process']['pid']
    before=protocol.budget(args.run,1024**3);gpu_before=protocol.gpu_set([pid]);b=protocol.metrics()
    port=b.endpoint(args.endpoint).port
    status,headers,_=gateway.complete_request(port,'/ready')
    if status!=200 or headers.get('x-aegis-voice-profile')!=args.profile:raise ValueError('Actual service not ready with expected profile')
    # Mandatory boot warmups call the engine directly. Any HTTP request before
    # this counter check destroys attribution of the first external request.
    if request_count(port)!=0:raise ValueError('Fresh process already served external requests; restart before benchmark')
    manifest=dict(plan=plan,plan_sha256=protocol.sha(args.run/'plan.json'),owner=owner,startup_evidence_sha256=protocol.sha(args.startup_evidence),startup=startup,
                  preflight=before,gpu_before=gpu_before,status='warming',cold_scope='Spawn→ready measured by launcher. First external request is warmups[0], never added to independent timings.')
    protocol.save(args.run/'manifest.json',manifest);rows=[];jobs=[];warmups=[];error=None
    from gateway.aegis_speech.experimental_text import normalizer_for
    normalize=normalizer_for(args.frontend)
    def call(case,rep,warmup=False):
        pcm,row=b.receive(args.endpoint,dict(input=case['text'],stream=True,response_format='pcm'),args.timeout)
        enforce_deadline(row,args.timeout)
        row.update(candidate=args.candidate,case=case['id'],size=case.get('size','warmup'),repetition=rep,
                   condition='first-external-after-ready' if warmup and rep==0 else 'common-external-warmup' if warmup else 'warm',
                   final_test=args.final_test,source_text=case['text'],effective_text=normalize.normalize(case['text']).normalized_text,
                   seed=protocol.SEED,seed_scope='Fixed in actual bound gateway runtime; no payload seed override',
                   scope=plan['settings']['scope'])
        if row['outcome']=='ok':
            try:row['gateway_completion']=completed_contract(args,row)
            except Exception as exc:row.update(outcome='failed_gateway_completion',error=repr(exc))
        if pcm and len(pcm)%2==0 and not warmup and (rep==0 or row['outcome']!='ok'):
            protocol.output_guard(args.run,len(pcm)+4096)
            path=args.run/f"{case['id']}-r{rep:02d}.wav";protocol.write_pcm(path,pcm);row.update(wav=path.name,wav_sha256=protocol.sha(path))
        protocol.output_guard(args.run);return row
    sampler=None
    try:
        with b.GpuSampler(args.run/'gpu.jsonl',True) as sampler:
            for index in range(3):
                warmups.append(call(dict(id='common_warmup',text=protocol.WARMUP),index,True));protocol.save(args.run/'warmups.json',warmups)
                if warmups[-1]['outcome']!='ok':raise RuntimeError('Common external warmup failed')
            manifest['status']='benchmarking';protocol.save(args.run/'manifest.json',manifest)
            with (args.run/'results.jsonl').open('x') as stream:
                for rep in range(args.repetitions):
                    for case in plan['cohort']['cases']:
                        protocol.gpu_set([pid]);row=call(case,rep);rows.append(row);protocol.append_row(stream,row)
                        job=protocol.asr_job(row,case,plan['cohort']['spoken'][case['id']],args.run)
                        if job:jobs.append(job)
                        protocol.save(args.run/'summary.json',b.summarize(rows))
            protocol.assert_matrix(rows,plan['cohort'],args.repetitions)
            if request_count(port)!=3+len(rows):raise RuntimeError('Concurrent/unaccounted external HTTP requests detected')
        if sampler.error:raise RuntimeError('GPU sampler failed: '+sampler.error)
        protocol.save(args.run/'gpu-overlap-audit.json',protocol.audit_gpu_samples(args.run/'gpu.jsonl',[pid]))
    except BaseException as exc:error=repr(exc);raise
    finally:
        same_sources=protocol.source_bindings(plan['source_hashes'])==plan['source_hashes']
        same_startup=protocol.sha(args.startup_evidence)==manifest['startup_evidence_sha256'];binding_error=None
        try:verified_startup(args)
        except Exception as exc:binding_error=repr(exc)
        final_budget_error=None
        try:final_budget=protocol.budget(args.run,0)
        except Exception as exc:final_budget_error=repr(exc);final_budget=dict(error=final_budget_error)
        protocol.save(args.run/'asr-jobs.json',dict(jobs=jobs,expected_conditions=plan['cohort']['case_denominator'],filtered=False))
        protocol.save(args.run/'execution.json',dict(completed_utc=protocol.utc(),expected_observations=plan['cohort']['case_denominator']*args.repetitions,
                      observations=len(rows),successes=sum(r['outcome']=='ok' for r in rows),error=error or final_budget_error,final_budget_error=final_budget_error,
                      sources_unchanged=same_sources,startup_receipt_unchanged=same_startup,final_live_binding_error=binding_error,
                      sampled_vram_peak_used_mib=sampler.peak if sampler else None,
                      final_working=final_budget,service_left_running=True,
                      lifecycle_scope='Client did not launch/stop server; launcher must verify shutdown or persistent handoff separately'))
        if not same_sources or not same_startup or binding_error:raise RuntimeError('Source/startup/live artifacts changed; evidence invalid')
        if final_budget_error and error is None:raise RuntimeError('Final budget guard failed; execution invalid, receipt preserved')

def main():
    parser=argparse.ArgumentParser(description=__doc__);protocol.common_arguments(parser)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8002/v1/audio/speech')
    parser.add_argument('--profile',required=True);parser.add_argument('--frontend',choices=('original-nfc','f5-refinement-1','ptbr-numbers-v1'),default='ptbr-numbers-v1')
    parser.add_argument('--model-sha256',required=True);parser.add_argument('--startup-evidence',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=180.)
    args=parser.parse_args();os.umask(0o077)
    plan=protocol.make_plan(args,'voxcpm2-http',settings(args),[Path(__file__),Path(gateway.__file__),*[protocol.REPO/p for p in gateway.RUNTIME_FILES],
        protocol.REPO/'gateway/aegis_speech/experimental_text.py',protocol.REPO/'gateway/aegis_speech/experimental_refinement.py',
        protocol.REPO/'gateway/aegis_speech/normalization.py',protocol.REPO/'gateway/aegis_speech/segmentation.py'])
    if protocol.plan_or_verify(args,plan):execute(args,plan)
    print(json.dumps(dict(status='executed' if args.action=='execute' else 'prepared_only',run=str(args.run),gpu_executed=args.action=='execute')))

if __name__=='__main__':main()
