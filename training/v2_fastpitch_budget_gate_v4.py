"""Future scoped-cache resume gate; default only prepares a private plan.

Preserves the pinned V3 scientific/checkpoint audit and adds explicit owner
authorization plus external cache/Docker ledger consumption before active.
No tensors/ASR/test-cohort access or Docker launch on import/plan.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from training import v2_delivery_protocol as protocol
from training import v2_qwen_delivery_benchmark as qwen
from training import v2_qwen_cache_client as cache
from training import v2_qwen_cache_audit_reader as reader

ROOT = qwen.ROOT
V3_SOURCE = ROOT / 'operations/fastpitch-budget-preflight-v3.py'
V3_SHA = '327345ea6b4e9fc25c331ee73f41be4bd2fc4c574a005c2b9734b459dea74b09'
CUDA_CACHE = ROOT / 'fastpitch/cuda-cache-main-v1'
CUDA_MAX = 16 * 1024**2
CUDA_RESERVE = 18 * 1024**2


def load_v3():
    if protocol.sha(V3_SOURCE) != V3_SHA: raise RuntimeError('Historical V3 source pin changed')
    spec = importlib.util.spec_from_file_location('aegis_immutable_resume_v3', V3_SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def review_guard(args):
    if not args.gpu_authorized or args.review is None: raise RuntimeError('Explicit reviewed V4 execution required')
    path = args.review
    if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise RuntimeError('Private owned review required')
    value = protocol.read(path)
    if (value.get('status') != 'approved_for_fastpitch_resume_v4'
            or value.get('gate_sha256') != protocol.sha(__file__)
            or value.get('plan_sha256') != protocol.sha(args.run / 'plan.json')
            or value.get('consume_external_and_docker_ledger') is not True):
        raise RuntimeError('Exact V4 source/plan/ledger approval required')
    return value


def authorization(owner, baseline_path):
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    if (value.get('container_id') != qwen.CONTAINER or value.get('owner_authorized') is not True
            or value.get('production_stopped') is not True):
        raise RuntimeError('V4 requires explicit production-stopped owner authorization')
    status = value.get('status')
    if status == 'active':
        if value.get('exclusive_aegisvoice_v2_gpu') is not True: raise RuntimeError('Active exclusivity differs')
        return value, None
    if status != 'budget_reconciliation_required' or value.get('exclusive_aegisvoice_v2_gpu') is not False:
        raise RuntimeError('Not an active or stopped pending-reconciliation window')
    prior_v4 = value.get('budget_v4_pending')
    if prior_v4 and not value.get('budget_reconciliation'):
        if (prior_v4.get('worker_pid') != owner['worker']['pid'] or prior_v4.get('owner_pid') != owner['guard_owner']['pid']
                or prior_v4.get('gate_sha256') != protocol.sha(__file__)):
            raise RuntimeError('Previous V4 failed admission belongs to another source/owner')
        return value, None
    pending = value.get('budget_reconciliation', {}); claim = value.get('qwen_baseline', {})
    if (claim.get('worker_pid') != owner['worker']['pid'] or claim.get('owner_pid') != owner['guard_owner']['pid']
            or pending.get('qwen_evidence_complete') is not True
            or pending.get('executor_sha256') != protocol.sha(qwen.__file__)
            or pending.get('cache_baseline') != str(baseline_path)
            or pending.get('cache_baseline_sha256') != protocol.sha(baseline_path)):
        raise RuntimeError('Pending Qwen evidence/owner/baseline differs')
    run = Path(pending['run'])
    if not run.is_relative_to(qwen.PREPARATION) or run.is_symlink(): raise RuntimeError('Pending Qwen run outside private scope')
    supplement = Path(pending['supplement_path'])
    if supplement != run / 'resume-budget-supplement.json' or protocol.sha(supplement) != pending['supplement_sha256']:
        raise RuntimeError('Pending supplement SHA/path differs')
    if protocol.sha(run / 'plan.json') != pending['plan_sha256']:
        raise RuntimeError('Pending Qwen plan changed')
    outcome = protocol.read(run / 'execution.json')
    cleanup = outcome.get('cleanup', {})
    if (outcome.get('cleanup_error') is not None or outcome.get('sources_unchanged') is not True
            or cleanup.get('qwen_evidence_complete') is not True or cleanup.get('stopped') is not True
            or cleanup.get('gpu_empty') is not True):
        raise RuntimeError('Qwen cleanup is not complete; no automatic recovery admission')
    # Generation/quality may fail while resource cleanup remains fully proven.
    return value, dict(supplement_path=str(supplement), supplement_sha256=protocol.sha(supplement),
                       qwen_run=str(run), qwen_execution_sha256=protocol.sha(run / 'execution.json'),
                       qwen_measurement_status=outcome.get('status'))


def evaluate(args, plan):
    review_guard(args)
    if args.cache_baseline is None or args.cache_baseline.resolve() != qwen.CACHE_BASELINE.resolve():
        raise RuntimeError('Canonical immutable goal-cutoff baseline required')
    if args.external_ledger is None: raise RuntimeError('Reviewed external NV/cache/tmp ledger required')
    v3 = load_v3()
    owner = qwen.owner_identity(check_cpu=False)
    expected_worker = v3.worker_guard(); v3.gpu_empty(); qwen.original(running=False)
    initial, consumed_qwen = authorization(owner, args.cache_baseline)
    if (os.environ.get('CUDA_CACHE_PATH') != str(CUDA_CACHE)
            or os.environ.get('CUDA_CACHE_MAXSIZE') != str(CUDA_MAX)):
        raise RuntimeError('Root-scoped16MiB CUDA cache environment required; scientific argv unchanged')
    if protocol.source_bindings(plan['source_hashes']) != plan['source_hashes']:
        raise RuntimeError('V4 plan sources changed')
    restored = (initial.get('budget_reconciliation', {}).get('original_experimental_authorization')
                or initial.get('budget_v4_pending', {}).get('original_authorization') or initial)
    if (restored.get('container_id') != qwen.CONTAINER or restored.get('owner_authorized') is not True
            or restored.get('production_stopped') is not True or restored.get('status') != 'active'
            or restored.get('exclusive_aegisvoice_v2_gpu') is not True):
        raise RuntimeError('Original experimental authorization is not coherent')
    closed = dict(initial, status='budget_reconciliation_required', exclusive_aegisvoice_v2_gpu=False,
                  budget_v4_pending=dict(gate_sha256=protocol.sha(__file__), plan_sha256=protocol.sha(args.run / 'plan.json'),
                    worker_pid=owner['worker']['pid'], owner_pid=owner['guard_owner']['pid'],
                    original_authorization=restored, run=str(args.run)))
    protocol.save(ROOT / 'gpu-window-authorized.json', closed)
    baseline = protocol.read(args.cache_baseline)
    # Historical V3 has parent/anchor/pause guards. V4 explicitly supplies auth
    # and the outside-core ledger that publication097f previously added itself.
    original = v3.run_gate(plan['command'], check_only=True)
    budget = qwen.phase_budget(args.run, qwen.original(running=False), 0, baseline, owner=owner,
                              external_ledger=args.external_ledger, core_usage=original['inventory'])
    usage = dict(original['inventory'], allocated_bytes=budget['total_accounted_bytes'])
    reserved = v3.plan_budget(usage, budget['free_bytes'], original['checkpoint_files'])
    # Retain root097f's18MiB payload+metadata margin even if already in core.
    reserved['projected_peak_bytes'] += CUDA_RESERVE
    reserved['extra_peak_bytes'] += CUDA_RESERVE
    reserved['cuda_cache_future_reserve_bytes'] = CUDA_RESERVE
    reserved['allowed'] = (reserved['allowed'] and reserved['projected_peak_bytes'] <= v3.LIMIT
                            and budget['free_bytes'] >= v3.FREE_RESERVE + reserved['extra_peak_bytes'])
    result = dict(status='fresh_v4_pass' if reserved['allowed'] else 'not_started_insufficient_peak_reserve',
                  allowed=reserved['allowed'], utc=protocol.utc(), plan_sha256=protocol.sha(args.run / 'plan.json'),
                  gate_sha256=protocol.sha(__file__), historical_v3_sha256=V3_SHA, owner=owner,
                  six_state_plan=reserved, current_ledger=budget, historical_scientific_audit=original,
                  consumed_qwen=consumed_qwen, consumed_external_specification_sha256=protocol.sha(args.external_ledger),
                  consumed_cache_baseline_sha256=protocol.sha(args.cache_baseline), training_not_started_at_reconciliation=True)
    protocol.save(args.run / 'reconciliation.json', result)
    if not result['allowed']: raise RuntimeError('V4 peak budget refused; authorization remains closed/unchanged')
    if (not qwen.same_owner(owner) or v3.worker_guard() != expected_worker
            or protocol.read(ROOT / 'gpu-window-authorized.json') != closed):
        raise RuntimeError('Owner/window/authorization changed during reconciliation')
    v3.gpu_empty(); qwen.original(running=False)
    if protocol.source_bindings(plan['source_hashes']) != plan['source_hashes']:
        raise RuntimeError('V4 source/baseline/specification changed while auditing')
    active = dict(restored, status='active', exclusive_aegisvoice_v2_gpu=True, production_stopped=True,
                  v4_ledger_consumed=dict(path=str(args.run / 'reconciliation.json'),
                    sha256=protocol.sha(args.run / 'reconciliation.json'), plan_sha256=result['plan_sha256'],
                    external_specification_sha256=result['consumed_external_specification_sha256'],
                    cache_baseline_sha256=result['consumed_cache_baseline_sha256']))
    protocol.save(ROOT / 'gpu-window-authorized.json', active)
    protocol.save(args.run / 'admission.json', dict(status='ledger_consumed_active_for_exact_command',
                  authorization_sha256=protocol.sha(ROOT / 'gpu-window-authorized.json'), command=plan['command']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('plan', 'execute'), default='plan')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--command-json', type=Path, default=ROOT / 'operations/096-fastpitch-original-job.json')
    parser.add_argument('--cache-baseline', type=Path)
    parser.add_argument('--external-ledger', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--gpu-authorized', action='store_true')
    args = parser.parse_args(); os.umask(0o077)
    if args.action == 'execute': review_guard(args)
    command = protocol.read(args.command_json)['command']
    sources = [Path(__file__), Path(qwen.__file__), Path(cache.__file__), Path(reader.__file__),
               Path(protocol.__file__), V3_SOURCE, cache.CHARGE_PATH, cache.PUBLICATION_PATH, args.command_json]
    sources += [path for path in (args.cache_baseline, args.external_ledger) if path is not None]
    plan = dict(status='prepared_only', created_utc=protocol.utc(), command=command,
        source_hashes=protocol.source_bindings(sources), v3_sha256=V3_SHA,
        cache_baseline_sha256=protocol.sha(args.cache_baseline) if args.cache_baseline else None,
        external_ledger_sha256=protocol.sha(args.external_ledger) if args.external_ledger else None,
        published_vllm_charge=cache.external_charge(), expected_cuda_environment=dict(CUDA_CACHE_PATH=str(CUDA_CACHE), CUDA_CACHE_MAXSIZE=str(CUDA_MAX)),
        policy='Explicit auth+fresh six-state V3 audit+outside cache fees+Docker ledger before active; no V3 source edit',
        tensors_loaded=False, actual_training_started=False)
    if protocol.plan_or_verify(args, plan):
        evaluate(args, plan)
        os.execv(command[0], command)
    print(json.dumps(dict(status='prepared_only', run=str(args.run))))


if __name__ == '__main__': main()
