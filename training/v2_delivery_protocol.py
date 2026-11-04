"""Shared delivery protocol. Import/plan use stdlib only; final reads are gated."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
HERE=REPO/'benchmarks/v2'
METRIC_SOURCE=HERE/'benchmark.py'
SEED=40733133
WARMUP='Esta é uma verificação de aquecimento.'
OUTPUT_CAP=512*1024**2

def metrics():
    spec=importlib.util.spec_from_file_location('aegis_delivery_metrics',METRIC_SOURCE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):metrics().atomic_json(path,value)
def sha(path):return metrics().digest(path)
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

def common_arguments(parser):
    parser.add_argument('--action',choices=('plan','execute'),default='plan')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--final-test',action='store_true')
    parser.add_argument('--decision',type=Path)
    parser.add_argument('--corpus',type=Path,help='Final only; explicit canonical test path after decision lock')
    parser.add_argument('--spoken',type=Path,help='Final only; canonical spoken transcription frozen before synthesis')
    parser.add_argument('--gpu-authorized',action='store_true')

def load_cohort(args):
    """The decision/roster guard must precede even a hash of the final corpus."""
    decision=None
    if args.final_test:
        decision=metrics().validate_final_decision(args.decision,args.candidate)
        if args.corpus is None or args.corpus.absolute()!=HERE/'test.pt-BR.json':
            raise ValueError('Final requires explicit canonical --corpus after decision lock')
        if args.spoken is None:raise ValueError('Final requires separately frozen --spoken before synthesis/ASR')
        corpus=args.corpus;spoken=args.spoken;expected=6
    else:
        if args.corpus is not None or args.spoken is not None or args.decision is not None:
            raise ValueError('Validation uses only the canonical validation7; alternate corpus/decision forbidden')
        corpus=HERE/'validation.pt-BR.json';spoken=HERE/'spoken-validation.pt-BR.json';expected=7
    cases=read(corpus);spoken_cases=read(spoken)['cases']
    if not isinstance(cases,list) or len(cases)!=expected:raise ValueError('Wrong complete cohort denominator')
    ids=[c.get('id') for c in cases]
    if len(set(ids))!=expected or any(not isinstance(i,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',i) for i in ids):
        raise ValueError('Unsafe/duplicate case IDs')
    if any(not isinstance(c.get('text'),str) or not c['text'].strip() or not isinstance(c.get('size'),str) for c in cases):
        raise ValueError('Missing text/size')
    if set(spoken_cases)!=set(ids) or any(not isinstance(v,str) or not v.strip() for v in spoken_cases.values()):
        raise ValueError('Spoken map must exactly cover the complete cohort')
    if any(c['text']==WARMUP for c in cases):raise ValueError('Warmup must be outside corpus')
    return dict(cases=cases,spoken=spoken_cases,case_denominator=expected,corpus=str(corpus),corpus_sha256=sha(corpus),
                spoken_path=str(spoken),spoken_sha256=sha(spoken),final_test=args.final_test,
                decision_sha256=sha(args.decision) if decision else None,
                decision_selected_candidate=decision['selected_candidate'] if decision else None,
                comparison_role=next((r['role'] for r in decision.get('comparison_roster',[]) if r['candidate']==args.candidate),None) if decision else None)

def source_bindings(paths):return {str(p):sha(p) for p in sorted(set(map(Path,paths)))}

def make_plan(args,backend,settings,sources):
    if args.repetitions<3:raise ValueError('At least three repetitions per case are required')
    cohort=load_cohort(args)
    return dict(status='prepared_only',created_utc=utc(),backend=backend,candidate=args.candidate,
                repetitions=args.repetitions,cohort=cohort,settings=settings,seed=SEED,seed_policy='fixed per call; never increment with repetition',
                common_external_warmups=dict(count=3,text=WARMUP,per_case=0,first='first request after readiness/load, recorded separately'),
                source_hashes=source_bindings([Path(__file__),HERE/'benchmark.py',*sources]),onset_rule=metrics().RULE,
                sample_rate=24000,format='PCM16LE mono',concurrency=1,
                gpu_executed=False,service_launched=False,torch_imported='torch' in sys.modules,
                final_scope='Common6 only. No target50 support; no selection or ASR execution.')

def plan_or_verify(args,plan):
    path=args.run.expanduser().resolve()
    if any((q/'.git').exists() for q in (path,*path.parents)):raise ValueError('Private artifacts must stay outside Git')
    if args.action=='plan':
        path.mkdir(parents=True,mode=0o700,exist_ok=False);args.run=path;save(path/'plan.json',plan);return False
    if not args.gpu_authorized:raise ValueError('Explicit root GPU authorization required')
    if not path.is_dir() or path.is_symlink() or path.stat().st_uid!=os.getuid():raise ValueError('Unsafe prepared run')
    previous=read(path/'plan.json')
    comparable=lambda value:{k:v for k,v in value.items() if k!='created_utc'}
    if comparable(previous)!=comparable(plan):raise ValueError('Frozen plan differs; preserve run and prepare a new plan')
    plan['created_utc']=previous['created_utc']
    if any((path/name).exists() for name in ('execution.json','manifest.json','results.jsonl','warmups.json')):
        raise ValueError('Fresh prepared run required after any execution')
    args.run=path;return True

def identity(pid):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    boot_seconds=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    start_utc=dt.datetime.fromtimestamp(boot_seconds+int(fields[19])/os.sysconf('SC_CLK_TCK'),dt.timezone.utc).isoformat()
    return dict(pid=int(pid),ppid=int(fields[1]),start_ticks=fields[19],uid=p.stat().st_uid,start_utc_estimated=start_utc,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                command=[v for v in (p/'cmdline').read_bytes().decode(errors='replace').split('\0') if v])

def owner_guard():
    me=identity(os.getpid());parent=identity(os.getppid());anchor=read(ROOT/'operations/gpu-window.json');auth=read(ROOT/'gpu-window-authorized.json')
    command=parent['command']
    def flag(key):
        try:return command[command.index(key)+1]
        except (ValueError,IndexError):return None
    if me['uid']!=os.getuid() or parent['uid']!=os.getuid() or flag('-m')!='training.v2_gpu_worker' or flag('--root')!=str(ROOT):
        raise RuntimeError('Must be an immediate child of the actual owned GPU worker')
    if anchor.get('child_pid')!=parent['pid'] or str(anchor.get('child_start'))!=parent['start_ticks']:
        raise RuntimeError('Worker PID/start differs from owner window')
    if auth.get('status')!='active' or any(auth.get(k) is not True for k in ('owner_authorized','production_stopped','exclusive_aegisvoice_v2_gpu')):
        raise RuntimeError('Exclusive owner GPU window inactive')
    if not(ROOT/'pause-asr').exists():raise RuntimeError('ASR/CPU performance pause required')
    from training.v2_voxcpm_optimized_probe import require_idle_cpu
    require_idle_cpu();return dict(self=me,worker=parent)

def gpu_set(expected):
    snap=metrics().gpu_snapshot();actual={int(p[0]) for p in snap['processes']}
    if actual!=set(expected):raise RuntimeError('GPU process set differs: '+str(actual))
    return snap

def budget(run,extra):
    from training.v2_language_monitor import physical_usage
    common=Path(subprocess.check_output(['git','-C',str(REPO),'rev-parse','--git-common-dir'],text=True).strip())
    if not common.is_absolute():common=REPO/common
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache'),common.resolve()]
    usage=physical_usage(roots);free=shutil.disk_usage(run).free
    if usage['permission_errors'] or usage['allocated_bytes']+extra>40_000_000_000 or free<5*1024**3+extra:
        raise RuntimeError('40GB working budget/5GiB volume reserve; reduce retained work before launch')
    return dict(working=usage,extra_reserved_bytes=extra,free_bytes=free,git_common=str(common.resolve()),
                docker_delta_scope='No Docker mutation by executor; launcher must separately account new Docker writable/log deltas')

def output_guard(run,extra_bytes=0):
    size=sum(p.lstat().st_blocks*512 for p in run.rglob('*') if p.is_file())
    if size+extra_bytes>OUTPUT_CAP or shutil.disk_usage(run).free<5*1024**3+extra_bytes:raise RuntimeError('Output512MiB/free5GiB limit')
    return size

def audit_gpu_samples(path,expected):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows or any({int(p[0]) for p in row['processes']}-set(expected) for row in rows):
        raise RuntimeError('GPU sampling absent or another process overlapped the measured run')
    return dict(samples=len(rows),allowed_pids=sorted(expected),limitation='500ms sampling may miss shorter overlaps/peaks')

def write_pcm(path,pcm):
    with wave.open(str(path),'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)

def asr_job(row,case,spoken,run):
    if row['repetition']!=0:return None
    job=dict(candidate=row['candidate'],case=case['id'],expected_verbatim=case['text'],expected_spoken=spoken,
             generation_outcome=row['outcome'],cohort='common-final6-rep0' if row['final_test'] else 'common-validation7-rep0')
    if row.get('wav'):
        path=Path(row['wav']);path=path if path.is_absolute() else run/path
        job.update(wav=str(path),input_audio_sha256=row['wav_sha256'])
    if row.get('effective_text') is not None:
        value=row['effective_text'];job['expected_effective']=' '.join(value) if isinstance(value,list) else value
    return job

def append_row(stream,row):
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def assert_matrix(rows,cohort,repetitions):
    wanted={(c['id'],r) for r in range(repetitions) for c in cohort['cases']}
    observed=[(r['case'],r['repetition']) for r in rows]
    if len(observed)!=len(wanted) or set(observed)!=wanted:raise ValueError('Incomplete/duplicate matrix; preserve full denominator')

