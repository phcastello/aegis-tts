"""Original Qwen transaction contracts, synthetic metadata/PCM and mocks only."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch
from contextlib import ExitStack

from training import v2_qwen_delivery_benchmark as q


class FakeSampler:
    def __init__(self, *args):
        self.error = None
        self.peak = 123.0
        self.samples = 5
    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeSafety:
    def __init__(self, *args): self.error = None
    def __enter__(self): return self
    def __exit__(self, *args): pass


class QwenDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.args = SimpleNamespace(run=self.run, candidate='qwen-warm', repetitions=3,
            final_test=False, timeout=180.0, gpu_authorized=True, review=self.root / 'review.json',
            cache_baseline=self.root / 'cache-original-baseline.json', external_ledger=self.root/'external-ledger.json')
        q.protocol.save(self.args.review, {})
        self.cache = self.root / 'cache'; self.cache.mkdir()
        self.baseline = dict(schema=q.cache_reader.SCHEMA, complete=True, root='/input-cache', cutoff_unix_ns=q.cache_reader.GOAL_START_NS, hashes_requested=True, entries={}, physical_bytes=0, fd_anchored=True, metadata_roster_passes=2)
        q.protocol.save(self.args.cache_baseline, self.baseline)
        q.protocol.save(self.run / 'model-before.json', {})
        self.cases = [dict(id='synthetic-' + str(i), text='Frase sintética ' + str(i) + '.', size='short') for i in range(7)]
        self.plan = dict(candidate='qwen-warm', repetitions=3, source_hashes={},
            cohort=dict(cases=self.cases, case_denominator=7, final_test=False,
                        spoken={case['id']: case['text'] for case in self.cases}),
            settings=dict(production_code_sha256={}))
        q.protocol.save(self.run / 'plan.json', self.plan)
        self.saved_auth = dict(status='active', production_stopped=True, owner_authorized=True,
            exclusive_aegisvoice_v2_gpu=True, container_id=q.CONTAINER, restore_only_after_all_gpu_phases=True)
        q.protocol.save(self.root / 'gpu-window-authorized.json', self.saved_auth)
        self.owner = dict(self=dict(pid=100, ppid=90, start_ticks='1', uid=os.getuid(), boot_id='boot'),
            worker=dict(pid=90, ppid=80, start_ticks='2', uid=os.getuid(), boot_id='boot'),
            guard_owner=dict(pid=80, ppid=1, start_ticks='3', uid=os.getuid(), boot_id='boot'))
        self.headers = {'x-aegis-tts-runtime-variant': q.RUNTIME, 'x-aegis-tts-dsp-variant': 'stable',
            'x-aegis-voice-profile': q.PROFILE, 'x-aegis-sample-rate': '24000', 'x-aegis-channels': '1',
            'x-aegis-request-id': 'synthetic-id'}

    def phase_file(self):
        with patch.object(q, 'ROOT', self.root):
            q.phase_write(self.saved_auth, self.owner, 'qwen_baseline_starting')
            q.phase_write(self.saved_auth, self.owner, 'qwen_baseline_active')

    def test_module_import_help_do_not_load_neural_or_mutate_docker(self):
        command = [sys.executable, '-c', "import sys; from training import v2_qwen_delivery_benchmark; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"]
        result = subprocess.run(command, cwd=q.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-m', 'training.v2_qwen_delivery_benchmark', '--help'], cwd=q.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--review', result.stdout)
        self.assertNotIn('--seed', result.stdout)

    def test_execute_review_is_required_before_any_cohort_read(self):
        argv = ['qwen', '--action', 'execute', '--run', str(self.run), '--candidate', 'qwen-warm', '--final-test']
        with patch.object(sys, 'argv', argv), patch.object(q.protocol, 'make_plan', side_effect=AssertionError('cohort touched')):
            with self.assertRaisesRegex(RuntimeError, 'review'):
                q.main()

    def test_shared_final_guard_rejects_before_closed_read_or_hash(self):
        args = SimpleNamespace(final_test=True, candidate='qwen-warm', decision=None,
            corpus=q.protocol.HERE / 'test.pt-BR.json', spoken=self.root / 'never-opened')
        with patch.object(q.protocol, 'read', side_effect=AssertionError('closed read')), patch.object(q.protocol, 'sha', side_effect=AssertionError('closed hash')):
            with self.assertRaises(ValueError):
                q.protocol.load_cohort(args)

    def test_review_binds_exact_source_plan_image_and_budget_policy(self):
        value = dict(status='approved_for_qwen_baseline', container_id=q.CONTAINER, image_id=q.IMAGE,
            executor_sha256=q.protocol.sha(q.__file__), plan_sha256=q.protocol.sha(self.run / 'plan.json'),
            cache_budget_policy='frozen-existing-cache-exclusion-new-or-changed-files-counted', resume_supplemental_budget_required=True)
        q.protocol.save(self.args.review, value)
        q.review_guard(self.args)
        for key in ('status', 'container_id', 'image_id', 'executor_sha256', 'plan_sha256', 'cache_budget_policy', 'resume_supplemental_budget_required'):
            invalid = dict(value, **{key: None})
            q.protocol.save(self.args.review, invalid)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                q.review_guard(self.args)

    def test_qwen_readiness_does_not_require_speech_profile_or_vox_frontend(self):
        q.headers_contract({key: value for key, value in self.headers.items() if key.startswith('x-aegis-tts-')})
        q.headers_contract(self.headers, speech=True)
        for key in ('x-aegis-tts-runtime-variant', 'x-aegis-tts-dsp-variant', 'x-aegis-voice-profile', 'x-aegis-sample-rate', 'x-aegis-channels'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.headers_contract(dict(self.headers, **{key: 'wrong'}), speech=True)

    def test_cgroup_requires_full_container_scope_and_accepts_actor_subgroups(self):
        path = '/system.slice/docker-' + q.CONTAINER + '.scope'
        anchor = q.parse_cgroups('0::' + path + '\n')
        self.assertTrue(q.member_cgroup(q.parse_cgroups('0::' + path + '/stage0\n'), anchor))
        for actual, expected in [('0::/\n', anchor), ('0::/other\n', anchor), ('0::' + path + '-sibling\n', anchor), ('0::' + path + '\n', {( '0', ''): '/'})]:
            with self.subTest(actual=actual):
                self.assertFalse(q.member_cgroup(q.parse_cgroups(actual), expected))

    def test_gpu_actor_may_differ_from_docker_init_and_mixed_actor_fails(self):
        path = '/system.slice/docker-' + q.CONTAINER + '.scope'
        groups = q.parse_cgroups('0::' + path + '\n')
        init = dict(pid=123, start_ticks='init')
        def identity(pid): return dict(pid=pid, start_ticks='init' if pid == 123 else 'actor', uid=0)
        metric = SimpleNamespace(gpu_snapshot=lambda: dict(processes=[['456', '1000']], devices=[]))
        with patch.object(q, 'original', return_value=dict(State=dict(Pid=123))), patch.object(q.protocol, 'identity', side_effect=identity), patch.object(q, 'cgroups', return_value=groups), patch.object(q.protocol, 'metrics', return_value=metric):
            result = q.gpu_binding(init)
            self.assertIn('456', result['gpu_actors'])
            metric.gpu_snapshot = lambda: dict(processes=[['456', '1000'], ['999', '1']], devices=[])
            with patch.object(q, 'cgroups', side_effect=lambda pid: groups if pid != 999 else {('0', ''): '/other'}), self.assertRaises(RuntimeError):
                q.gpu_binding(init)

    def test_auth_truth_is_false_while_original_qwen_is_online(self):
        self.phase_file()
        phase = q.protocol.read(self.root / 'gpu-window-authorized.json')
        self.assertEqual(phase['status'], 'qwen_baseline_active')
        self.assertFalse(phase['production_stopped'])
        self.assertFalse(phase['exclusive_aegisvoice_v2_gpu'])
        self.assertTrue(phase['owner_authorized'])
        self.assertEqual(self.saved_auth['production_stopped'], True)

    def test_auth_transition_refuses_overwriting_another_phase(self):
        with patch.object(q, 'ROOT', self.root):
            q.protocol.save(self.root / 'gpu-window-authorized.json', dict(self.saved_auth, status='emergency_closed'))
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                q.phase_write(self.saved_auth, self.owner, 'qwen_baseline_starting')
            self.assertEqual(q.protocol.read(self.root / 'gpu-window-authorized.json')['status'], 'emergency_closed')

    def test_owner_loss_hands_recovery_to_watchdog_without_stop_or_auth_restore(self):
        with patch.object(q, 'same_owner', return_value=False), patch.object(q.subprocess, 'run', side_effect=AssertionError('Docker mutation')):
            result = q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
        self.assertEqual(result['status'], 'owner_lost_handed_to_watchdog')
        self.assertFalse(result['experimental_authorization_restored'])

    def test_cleanup_leaves_reconciliation_pending_after_actual_stop_empty_and_budget(self):
        self.phase_file()
        events = []
        def docker(command, **kwargs): events.append(('docker', command)); return SimpleNamespace(returncode=0)
        def gpu(expected): events.append(('gpu', expected)); return dict(processes=[])
        def budget(*args, **kwargs): events.append(('budget', 0)); return dict(total_accounted_bytes=1)
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'same_owner', return_value=True), patch.object(q, 'original', side_effect=[dict(State=dict(Running=True)), dict(State=dict(Running=False)), dict(State=dict(Running=False))]), patch.object(q.subprocess, 'run', side_effect=docker), patch.object(q.protocol, 'gpu_set', side_effect=gpu), patch.object(q, 'production_code', return_value={}), patch.object(q, 'model_contract', return_value={}), patch.object(q, 'phase_budget', side_effect=budget):
            result = q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                q.initial_auth()
        self.assertFalse(result['experimental_authorization_restored'])
        pending = q.protocol.read(self.root / 'gpu-window-authorized.json')
        self.assertEqual(pending['status'], 'budget_reconciliation_required')
        self.assertTrue(pending['production_stopped']); self.assertFalse(pending['exclusive_aegisvoice_v2_gpu'])
        self.assertEqual(pending['budget_reconciliation']['original_experimental_authorization'], self.saved_auth)
        self.assertTrue(pending['budget_reconciliation']['qwen_evidence_complete'])
        self.assertEqual(events[0], ('docker', ['docker', 'stop', '--time', '10', q.CONTAINER]))
        self.assertEqual([entry[0] for entry in events], ['docker', 'gpu', 'budget'])

    def test_cleanup_budget_failure_keeps_baseline_auth_closed(self):
        self.phase_file()
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'same_owner', return_value=True), patch.object(q, 'original', return_value=dict(State=dict(Running=False))), patch.object(q.protocol, 'gpu_set', return_value=dict(processes=[])), patch.object(q, 'production_code', return_value={}), patch.object(q, 'phase_budget', side_effect=RuntimeError('budget failed')):
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
        phase = q.protocol.read(self.root / 'gpu-window-authorized.json')
        self.assertTrue(phase['production_stopped'])
        self.assertFalse(phase['exclusive_aegisvoice_v2_gpu'])
        self.assertEqual(phase['status'], 'budget_reconciliation_required')
        self.assertFalse(phase['budget_reconciliation']['qwen_evidence_complete'])

    def test_phase_budget_includes_cache_git_and_docker_reserve(self):
        cache = self.cache
        info = dict(Mounts=[dict(Source=str(cache), Destination='/cache')])
        base = 40_000_000_000 - q.DOCKER_LOG_CAP - q.EXTRA_RESERVE - q.cache_client.CHARGE_FLOOR
        def usage(paths): return dict(allocated_bytes=base if len(paths) > 1 else 123, permission_errors=[])
        from training import v2_language_monitor as ledger
        with patch.object(ledger, 'physical_usage', side_effect=usage) as measured, patch.object(q.subprocess, 'check_output', return_value=str(self.root / 'git')), patch.object(q, 'docker_size', return_value=0), patch.object(q, 'capture_cache', return_value=self.baseline), patch.object(q.cache_client, 'extra_external', return_value=dict(accounted_bytes=0)), patch('shutil.disk_usage', return_value=SimpleNamespace(free=5 * 1024**3 + q.EXTRA_RESERVE)):
            result = q.phase_budget(self.run, info, q.EXTRA_RESERVE, self.baseline)
            self.assertNotIn(cache, measured.call_args_list[0].args[0])
            self.assertIn(self.root / 'git', measured.call_args_list[0].args[0])
            self.assertEqual(result['total_accounted_bytes'] + q.EXTRA_RESERVE, 40_000_000_000)
            with patch.object(q, 'docker_size', return_value=q.DOCKER_RW_CAP + 1), self.assertRaises(RuntimeError):
                q.phase_budget(self.run, info, q.EXTRA_RESERVE, self.baseline)

    def test_payload_fixed_no_seed_override_and_deadline_failure_preserved(self):
        calls = []
        pcm = b'\x00\x20' * 2400
        metric = SimpleNamespace(receive=lambda endpoint, payload, timeout: (calls.append((endpoint, payload)) or pcm,
            dict(outcome='ok', headers=self.headers, eof_ms=181000)))
        with patch.object(q.protocol, 'metrics', return_value=metric):
            row = q.call(self.args, self.cases[0], 0, True)
        self.assertEqual(row['outcome'], 'failed_deadline')
        self.assertEqual(calls, [(q.ENDPOINT, dict(input=self.cases[0]['text'], stream=True, response_format='pcm'))])
        self.assertNotIn('effective_text', row)

    def test_aborted_matrix_fills_every_case_and_rep_without_filtering(self):
        rows = [dict(candidate='qwen-warm', case=self.cases[0]['id'], repetition=0, outcome='failed')]
        completed = q.complete_matrix(rows, self.plan, 'abort')
        self.assertEqual(len(completed), 21)
        self.assertEqual(sum(row['outcome'] == 'not_attempted' for row in completed), 20)
        with self.assertRaises(ValueError): q.complete_matrix(rows + rows, self.plan)

    def test_cache_capture_requires_owned_cpu_reader_and_never_falls_back_to_host(self):
        info = dict(Mounts=[dict(Source=str(self.cache), Destination='/cache')])
        with self.assertRaisesRegex(RuntimeError,'no host fallback'): q.capture_cache(info)
        guard=lambda: None
        with patch.object(q.cache_reader,'HOST_CACHE',str(self.cache)), patch.object(q.cache_client,'quiet_snapshot',return_value=self.baseline) as capture:
            self.assertEqual(q.capture_cache(info,self.run/'audit',guard),self.baseline)
        capture.assert_called_once_with(self.run/'audit',guard)

    def test_published_external_cache_floor_is_not_rebaselined_away(self):
        self.assertEqual(q.cache_fee(self.baseline,self.baseline)['accounted_bytes'],q.cache_client.CHARGE_FLOOR)

    def test_freeze_refuses_busy_gpu_before_any_cache_payload_read(self):
        argv = ['qwen', '--freeze-cache', '--gpu-authorized', '--run', str(self.run), '--candidate', 'qwen-warm',
                '--cache-baseline', str(self.args.cache_baseline)]
        with patch.object(sys, 'argv', argv), patch.object(q, 'owner_identity', return_value=self.owner), patch.object(q, 'initial_auth', return_value=self.saved_auth), patch.object(q, 'original', return_value={}), patch.object(q.protocol, 'gpu_set', side_effect=RuntimeError('busy')), patch.object(q, 'capture_cache', side_effect=AssertionError('cache payload touched')):
            with self.assertRaisesRegex(RuntimeError, 'busy'): q.main()

    def test_model_metadata_only_does_not_hash_weights_and_real_guard_rejects_changed_payload(self):
        model = self.root / 'local-model'; model.mkdir()
        metadata = model / '.cache/huggingface/download'; metadata.mkdir(parents=True)
        payload = model / 'model.safetensors'; payload.write_bytes(b'synthetic weight, no tensor')
        expected = hashlib.sha256(payload.read_bytes()).hexdigest()
        (metadata / 'model.safetensors.metadata').write_text(q.MODEL_REVISION + '\n' + expected + '\n0\n')
        info = dict(Mounts=[dict(Destination='/models/Qwen3-TTS-12Hz-0.6B-Base', Source=str(model), RW=False)],
                    Config=dict(Env=['HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1']))
        digest = q.protocol.sha
        def metadata_only(path):
            if Path(path) == payload: raise AssertionError('weight payload hashed in plan')
            return digest(path)
        with patch.object(q, 'MODEL_ROOT', model), patch.object(q, 'MODEL_PINS', {'model.safetensors': expected}):
            with patch.object(q.protocol, 'sha', side_effect=metadata_only):
                value = q.model_contract(info, hash_payloads=False)
            self.assertFalse(value['artifacts'][0]['payload_hashed'])
            self.assertTrue(q.model_contract(info, hash_payloads=True)['artifacts'][0]['payload_hashed'])
            payload.write_bytes(b'changed synthetic weight')
            with self.assertRaisesRegex(RuntimeError, 'hash differs'):
                q.model_contract(info, hash_payloads=True)

    def test_model_online_flags_or_loader_precedence_drift_rejected(self):
        model = self.root / 'local-model'; model.mkdir()
        metadata = model / '.cache/huggingface/download'; metadata.mkdir(parents=True)
        payload = model / 'model.safetensors'; payload.write_bytes(b'synthetic')
        expected = hashlib.sha256(payload.read_bytes()).hexdigest()
        (metadata / 'model.safetensors.metadata').write_text(q.MODEL_REVISION + '\n' + expected + '\n0\n')
        info = dict(Mounts=[dict(Destination='/models/Qwen3-TTS-12Hz-0.6B-Base', Source=str(model), RW=False)],
                    Config=dict(Env=['HF_HUB_OFFLINE=0', 'TRANSFORMERS_OFFLINE=1']))
        with patch.object(q, 'MODEL_ROOT', model), patch.object(q, 'MODEL_PINS', {'model.safetensors': expected}):
            with self.assertRaisesRegex(RuntimeError, 'offline'): q.model_contract(info)
            info['Config']['Env'][0] = 'HF_HUB_OFFLINE=1'
            (model / 'pytorch_model.bin').write_bytes(b'alternate loader input')
            with self.assertRaisesRegex(RuntimeError, 'precedence'): q.model_contract(info)

    def test_sampler_records_live_actor_proof_and_reports_poll_failure(self):
        path = self.run / 'sample.jsonl'
        sampler = q.CgroupSampler(path, dict(pid=123, start_ticks='init'))
        snapshot = dict(snapshot=dict(devices=[['uuid', 'GPU', '12000', '321']], processes=[['456', '321']]),
                        gpu_actors={'456': dict(identity=dict(pid=456), cgroups=[])})
        with path.open('x') as stream, patch.object(q, 'gpu_binding', return_value=snapshot):
            sampler.capture(stream)
        self.assertEqual(sampler.samples, 1); self.assertEqual(sampler.peak, 321)
        other = q.CgroupSampler(self.run / 'failed-sample.jsonl', dict(pid=123))
        with patch.object(q, 'gpu_binding', side_effect=RuntimeError('foreign actor')):
            other.loop()
        self.assertIn('foreign actor', other.error)

    def test_completion_requires_matching_status_and_named_sse_events(self):
        row = dict(case='synthetic', repetition=0, headers=self.headers)
        status = (200, {}, b'{"request_id":"synthetic-id","status":"completed"}')
        with patch.object(q.wire, 'complete_request', side_effect=[status, (200, {}, b'event: accepted\n\nevent: completed\n')]):
            self.assertIn('sse', q.completion(row))
        with patch.object(q.wire, 'complete_request', side_effect=[status, (200, {}, b'event: accepted\n\nevent: failed\n\nevent: completed\n')]):
            with self.assertRaisesRegex(ValueError, 'SSE'): q.completion(row)

    def test_post_matrix_contracts_use_real_wav_pcm_sse_cancel_and_next_request(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(b'\x00\x20' * 2400)
        calls = []
        headers = dict(self.headers, **{'content-type': 'audio/wav'})
        def complete(port, route, method='GET', payload=None, timeout=30):
            calls.append((route, method))
            if method == 'POST': return 200, headers, buffer.getvalue()
            if method == 'DELETE': return 200, {}, b'{"status":"cancelling"}'
            if route.endswith('/cancel-id/events'): return 200, {}, b'event: accepted\n\nevent: cancelled\n'
            if route.endswith('/events'): return 200, {}, b'event: accepted\n\nevent: completed\n'
            if route.endswith('/cancel-id'): return 200, {}, b'{"status":"cancelled"}'
            if route.endswith('/synthetic-id'): return 200, {}, b'{"request_id":"synthetic-id","status":"completed"}'
            if route.endswith('/queue'): return 200, {}, b'{"active_request_id":null,"queued_requests":0,"queued_segments":0}'
            return 200, {}, b'{}'
        class Response:
            status = 200
            def getheaders(inner): return list(self.headers.items())
            def getheader(inner, name, default=None): return 'audio/pcm' if name.lower() == 'content-type' else 'cancel-id'
            def read1(inner, size): return b'\x00\x20' * 10
        connection = SimpleNamespace(close=lambda: calls.append(('closed', '')))
        metric = q.protocol.metrics()
        with patch.object(metric, 'receive', side_effect=lambda *args: (b'\x00\x20' * 2400, dict(outcome='ok', headers=self.headers, eof_ms=100))), patch.object(q.protocol, 'metrics', return_value=metric), patch.object(q.wire, 'complete_request', side_effect=complete), patch.object(q.wire, 'request', return_value=(connection, Response())):
            result = q.post_contracts(self.args)
        self.assertEqual(result['native_pcm']['condition'], 'post-matrix-native-contract')
        self.assertEqual(result['next_after_cancel']['condition'], 'post-matrix-next-after-cancel')
        self.assertEqual(result['cancel']['first_pcm_bytes'], 20)
        for route in ('/v1/audio/speech', '/v1/aegis/speech'):
            self.assertTrue(Path(result[route]['wav']).is_file())
            self.assertEqual(q.protocol.sha(result[route]['wav']), result[route]['wav_sha256'])
        self.assertIn(('/v1/aegis/speech/cancel-id', 'DELETE'), calls)
        self.assertEqual(calls[-3:], [('/health', 'GET'), ('/ready', 'GET'), ('/v1/aegis/queue', 'GET')])

    def test_safety_phase_deadline_requests_owned_cleanup_without_real_signal(self):
        watch = q.SafetyMonitor(self.run, dict(free_bytes=10 * 1024**3), {}, self.baseline, 50)
        watch.stop = SimpleNamespace(wait=lambda seconds: False)
        with patch.object(q.protocol, 'output_guard', return_value=0), patch.object(q.time, 'monotonic', return_value=100), patch.object(q.os, 'kill') as kill:
            watch.loop()
        self.assertIn('phase ceiling', watch.error)
        kill.assert_called_once_with(os.getpid(), q.signal.SIGTERM)
        self.assertTrue((self.run / 'safety-stop.json').exists())

    def cache_watcher_fixture(self):
        info=dict(Mounts=[dict(Source=str(self.cache),Destination='/cache')])
        path=self.cache/'cached-file';path.write_bytes(b'a'*4096)
        original=q.cache_reader.signature(path.stat())
        baseline=q.cache_reader.snapshot(self.cache,True)
        baseline['entries'][path.name]['metadata_signature'][3:5]=[q.cache_reader.GOAL_START_NS-1]*2
        baseline['entries'][path.name]['eligible_before_goal']=True
        fake=dict(kind='file',metadata_signature=[999,999,21049344,q.cache_reader.GOAL_START_NS+1,q.cache_reader.GOAL_START_NS+1,41112,1],device=999,inode=999,bytes=21049344,allocated_bytes=21049344,links=1,sha256='0'*64,uncertain_metadata=False,eligible_before_goal=False)
        baseline['entries']['published-post-goal']=fake
        initial=q.cache_fee(baseline,baseline)
        baseline['physical_bytes']=initial['physical_bytes']
        before=dict(free_bytes=20*1024**3,total_accounted_bytes=38_000_000_000,cache_physical_bytes=initial['physical_bytes'],cache_accounted_bytes=initial['accounted_bytes'])
        def capture():
            frame=q.cache_reader.snapshot(self.cache,False)
            if frame['entries'][path.name]['metadata_signature']==original:
                frame['entries'][path.name]['metadata_signature']=baseline['entries'][path.name]['metadata_signature'].copy()
            frame['entries']['published-post-goal']=dict(fake,sha256=None)
            frame['physical_bytes']=q.cache_fee(baseline,frame)['physical_bytes']
            return frame
        auditor=SimpleNamespace(snapshot=capture)
        watch=q.SafetyMonitor(self.run,before,info,baseline,q.time.monotonic()+1000,auditor)
        return info,path,baseline,before,watch

    def test_cache_metadata_watcher_charges_same_size_overwrite_without_free_loss_or_sha(self):
        info, path, baseline, before, watch = self.cache_watcher_fixture()
        path.write_bytes(b'b' * 4096)
        previous = baseline['entries'][path.name]['metadata_signature'][3]
        os.utime(path, ns=(previous + 1_000_000_000, previous + 1_000_000_000))
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=before['free_bytes'])), patch.object(q.protocol, 'sha', side_effect=AssertionError('Payload read during timing')):
            row = watch.measure()
        self.assertEqual(row['physical_free_loss_bytes'], 0)
        self.assertEqual(row['cache_physical_delta_bytes'], 0)
        self.assertEqual(row['cache_fee_growth_bytes'], path.stat().st_blocks * 512)
        self.assertEqual(row['estimated_total_bytes'], before['total_accounted_bytes'] + row['cache_fee_growth_bytes'])
        self.assertFalse(row['payloads_hashed'])

    def test_cache_metadata_watcher_stops_large_in_place_fee_even_with_constant_free_space(self):
        info, path, baseline, before, _ = self.cache_watcher_fixture()
        # Synthetic allocation metadata proves the2GiB case without allocating2GiB.
        old = baseline['entries'][path.name]
        old['allocated_bytes'] = 2 * 1024**3; old['metadata_signature'][-2] = old['allocated_bytes'] // 512
        current = copy.deepcopy(baseline); current['entries'][path.name]['metadata_signature'][3] += 1
        before['cache_physical_bytes'] = q.cache_fee(baseline,baseline)['physical_bytes']
        self.assertLess(before['total_accounted_bytes'] + q.EXTRA_RESERVE, 40_000_000_000)
        watch = q.SafetyMonitor(self.run, before, info, baseline, q.time.monotonic() + 1000, SimpleNamespace(snapshot=lambda:current))
        watch.stop = SimpleNamespace(wait=lambda seconds: False)
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=before['free_bytes'])), patch.object(q.protocol, 'output_guard', return_value=0), patch.object(q.os, 'kill') as kill:
            watch.loop()
        row = q.protocol.read(self.run / 'safety-metadata-latest.json')
        self.assertEqual(row['physical_free_loss_bytes'], 0)
        self.assertGreater(row['estimated_total_bytes'], 40_000_000_000)
        self.assertIn('cache fee', watch.error); kill.assert_called_once()

    def test_new_cache_physical_growth_is_counted_once(self):
        info, path, baseline, before, watch = self.cache_watcher_fixture()
        (self.cache / 'new-cache-file').write_bytes(b'new' * 4096)
        delta = q.cache_fee(baseline,watch.auditor.snapshot())['physical_bytes'] - before['cache_physical_bytes']
        self.assertGreater(delta, 0)
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=before['free_bytes'] - delta)):
            row = watch.measure()
        self.assertEqual(row['outside_cache_growth_bytes'], 0)
        self.assertEqual(row['cache_fee_growth_bytes'], delta)
        self.assertEqual(row['estimated_total_bytes'], before['total_accounted_bytes'] + delta)

    def test_metadata_watcher_that_does_not_join_cannot_qualify_timing(self):
        watch = q.SafetyMonitor(self.run, dict(free_bytes=10 * 1024**3), {}, self.baseline, 50)
        watch.thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
        watch.__exit__(None, None, None)
        self.assertIn('did not stop', watch.error)

    def test_original_inspector_has_bounded_docker_timeout_without_ops_inspector(self):
        timeout = subprocess.TimeoutExpired(['docker', 'inspect', q.CONTAINER], q.INSPECT_TIMEOUT)
        with patch.object(q.subprocess, 'check_output', side_effect=timeout) as inspect, patch.object(q.ops, 'inspect', side_effect=AssertionError('unbounded inspect')), patch.object(q.ops, 'assert_original', side_effect=AssertionError('unbounded assert')):
            with self.assertRaises(subprocess.TimeoutExpired): q.original()
        self.assertEqual(inspect.call_args.args[0], ['docker', 'inspect', q.CONTAINER])
        self.assertEqual(inspect.call_args.kwargs['timeout'], 10)

    def protected_fixture(self):
        protected = self.root / 'synthetic-protected'; protected.mkdir()
        files = [protected / ('artifact-%03d.dat' % index) for index in range(77)]
        for index, path in enumerate(files): path.write_bytes(('synthetic-' + str(index)).encode())
        files[0].unlink(); files[0].symlink_to(files[1].name)
        expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        info = dict(Id=q.CONTAINER, Image=q.IMAGE, Name='/synthetic-original', Config=dict(Env=[]), HostConfig={},
                    Mounts=[dict(Source=str(protected), Destination='/protected', RW=False)], State=dict(Running=False))
        directory = self.root / 'synthetic-ops'; directory.mkdir()
        q.protocol.save(directory / 'snapshot.json', info); q.protocol.save(directory / 'reference-hashes.json', expected)
        return files, info, directory

    def test_original_reference_proof_hashes_only_synthetic77_and_rejects_drift(self):
        files, info, directory = self.protected_fixture()
        with patch.object(q, 'OPS', directory), patch.object(q.subprocess, 'check_output', return_value=json.dumps([info])):
            proof = q.original(running=False, references=True)['_qwen_reference_proof']
            self.assertEqual(proof['artifact_count'], 77)
            files[-1].write_bytes(b'changed synthetic artifact')
            with self.assertRaisesRegex(RuntimeError, 'hashes changed'): q.original(references=True)

    def test_original_reference_roster_and_chunk_deadline_reject_before_payload(self):
        files, info, directory = self.protected_fixture()
        with patch.object(q, 'OPS', directory), patch.object(q.subprocess, 'check_output', return_value=json.dumps([info])):
            with patch.object(q.time, 'monotonic', side_effect=[0, 31]), self.assertRaisesRegex(TimeoutError, 'deadline'):
                q.original(references=True)
            (files[0].parent / 'unbound-artifact').write_bytes(b'synthetic unbound')
            with patch.object(q.hashlib, 'sha256', side_effect=AssertionError('Payload hashed before roster guard')), self.assertRaisesRegex(RuntimeError, 'roster'):
                q.original(references=True)

    def test_cleanup_inspect_timeout_uses_bounded_exact_stop_and_keeps_auth_pending(self):
        self.phase_file()
        timeout = subprocess.TimeoutExpired(['docker', 'inspect', q.CONTAINER], q.INSPECT_TIMEOUT)
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'same_owner', return_value=True), patch.object(q, 'original', side_effect=[timeout, dict(State=dict(Running=False)), dict(State=dict(Running=False))]), patch.object(q.subprocess, 'run') as docker, patch.object(q.protocol, 'gpu_set', return_value=dict(processes=[])), patch.object(q, 'production_code', return_value={}), patch.object(q, 'model_contract', return_value={}), patch.object(q, 'phase_budget', return_value={}):
            result = q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
        self.assertFalse(result['experimental_authorization_restored'])
        self.assertTrue((self.run / 'cleanup-inspect-timeout.json').exists())
        self.assertEqual(docker.call_args.args[0], ['docker', 'stop', '--time', '10', q.CONTAINER])
        self.assertEqual(docker.call_args.kwargs['timeout'], 20)
        self.assertEqual(q.protocol.read(self.root / 'gpu-window-authorized.json')['status'], 'budget_reconciliation_required')

    def test_owner_loss_after_stop_abandons_post_hashing_and_auth_restore(self):
        self.phase_file()
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'same_owner', side_effect=[True, False]), patch.object(q, 'original', return_value=dict(State=dict(Running=True))), patch.object(q.subprocess, 'run') as docker, patch.object(q, 'production_code', side_effect=AssertionError('post hashing')):
            result = q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
        self.assertEqual(result['status'], 'owner_lost_after_stop_handed_to_watchdog')
        docker.assert_called_once()
        self.assertEqual(docker.call_args.args[0], ['docker', 'stop', '--time', '10', q.CONTAINER])
        self.assertEqual(docker.call_args.kwargs['timeout'], 20)
        self.assertLess(docker.call_args.kwargs['timeout'], 30)
        self.assertFalse(result['reconciliation_complete'])
        self.assertFalse(q.protocol.read(self.root / 'gpu-window-authorized.json')['production_stopped'])

    def test_cleanup_never_reopens_auth_when_gpu_does_not_empty(self):
        self.phase_file()
        with patch.object(q, 'ROOT', self.root), patch.object(q, 'same_owner', return_value=True), patch.object(q, 'original', return_value=dict(State=dict(Running=False))), patch.object(q.protocol, 'gpu_set', side_effect=RuntimeError('still active')), patch.object(q.time, 'monotonic', side_effect=[0, 100]):
            with self.assertRaisesRegex(RuntimeError, 'did not empty'):
                q.cleanup_transaction(self.args, self.owner, self.saved_auth, True, self.plan)
        self.assertFalse(q.protocol.read(self.root / 'gpu-window-authorized.json')['production_stopped'])

    def test_mocked_full_transaction_has_three_common_warmups_then21_and_owned_stop(self):
        self.plan['settings']['production_code_sha256'] = {}
        q.protocol.save(self.run / 'plan.json', self.plan)
        self.phase_file()
        q.protocol.save(self.root / 'gpu-window-authorized.json', self.saved_auth)
        state = dict(running=False)
        calls, docker_commands = [], []
        def original(running=None, references=False):
            if running is not None and state['running'] is not running: raise RuntimeError('synthetic state mismatch')
            return dict(State=dict(Running=state['running'], Pid=123, StartedAt='synthetic'))
        def docker(command, **kwargs):
            docker_commands.append(command)
            if command[1] == 'start': state['running'] = True
            elif command[1] == 'stop': state['running'] = False
            else: raise AssertionError(command)
            return SimpleNamespace(returncode=0)
        def call(args, case, repetition, warmup=False, native=False):
            calls.append((case['id'], repetition, warmup))
            return dict(candidate=args.candidate, case=case['id'], size=case.get('size', 'warmup'),
                        repetition=repetition, condition='warm', final_test=False, outcome='ok')
        patches = [patch.object(q, 'ROOT', self.root), patch.object(q, 'review_guard', return_value={}),
            patch.object(q, 'owner_identity', return_value=self.owner), patch.object(q, 'same_owner', return_value=True),
            patch.object(q, 'original', side_effect=original), patch.object(q, 'initial_auth', return_value=self.saved_auth),
            patch.object(q, 'production_code', return_value={}), patch.object(q, 'phase_budget', return_value=dict(free_bytes=10 * 1024**3)),
            patch.object(q, 'capture_cache', return_value=self.baseline), patch.object(q, 'model_contract', return_value={}),
            patch.object(q, 'CACHE_BASELINE', self.args.cache_baseline),
            patch.object(q.protocol, 'gpu_set', return_value=dict(processes=[])), patch.object(q.protocol, 'identity', return_value=dict(pid=123, start_ticks='init')),
            patch.object(q, 'CgroupSampler', FakeSampler), patch.object(q, 'SafetyMonitor', FakeSafety),
            patch.object(q.cache_client, 'Auditor', return_value=SimpleNamespace(__enter__=lambda:None,close=lambda:None)),
            patch.object(q, 'baseline_guard', return_value={}), patch.object(q, 'request_count', side_effect=[0, 24]),
            patch.object(q, 'call', side_effect=call), patch.object(q, 'post_contracts', return_value=dict(synthetic=True)),
            patch.object(q.wire, 'complete_request', return_value=(200, self.headers, b'')), patch.object(q.subprocess, 'run', side_effect=docker)]
        with ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            result = q.execute(self.args, self.plan)
        self.assertEqual(result['status'], 'completed_pending_budget_reconciliation')
        self.assertEqual(len(calls), 24)
        self.assertEqual(calls[:3], [('common_warmup', i, True) for i in range(3)])
        self.assertEqual(docker_commands, [['docker', 'start', q.CONTAINER], ['docker', 'stop', '--time', '10', q.CONTAINER]])
        self.assertFalse(state['running'])
        self.assertEqual(q.protocol.read(self.root / 'gpu-window-authorized.json')['status'], 'budget_reconciliation_required')
        jobs = q.protocol.read(self.run / 'asr-jobs.json')
        self.assertEqual(len(jobs['jobs']), 7)
        self.assertEqual(result['expected_observations'], 21)


if __name__ == '__main__':
    unittest.main()
