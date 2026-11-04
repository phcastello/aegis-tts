"""Stdlib-only contracts for the isolated GPU quality runner; never load a model."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from training import v2_fastpitch_gpu_quality as quality


REPO = Path(__file__).resolve().parents[1]


class GpuQualityTests(unittest.TestCase):
    def identities(self):
        root = Path('/var/tmp/synthetic-aegis-quality')
        parent = dict(pid=1234, ppid=1, start_ticks='111', uid=os.getuid(),
                      command=['python3', '-m', 'training.v2_gpu_worker', '--root', str(root)])
        me = dict(pid=1235, ppid=1234, start_ticks='112', uid=os.getuid(),
                  command=['python3', '-m', 'training.v2_fastpitch_gpu_quality', '--execute'])
        anchor = dict(child_pid=1234, child_start='111')
        authorization = dict(owner_authorized=True, exclusive_aegisvoice_v2_gpu=True,
                             production_stopped=True, status='active')
        return me, parent, anchor, authorization, root

    def write_wave(self, path, *, channels=1, width=2, rate=24000, frames=32):
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(channels)
            output.setsampwidth(width)
            output.setframerate(rate)
            output.writeframes(b'\x01' * (frames * channels * width))

    def synthetic_rows(self, root):
        cases = [dict(id='synthetic-' + str(index)) for index in range(7)]
        rows = []
        for case in cases:
            path = root / (case['id'] + '.wav')
            self.write_wave(path)
            rows.append(dict(case=case['id'], outcome='ok', finite=True, peak=0.01,
                             clipped_samples=0, outside_unit=0, samples_at_generator_bound=0,
                             samples=32, audio_seconds=32 / 24000, wav=str(path),
                             wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return rows, cases

    def test_only_the_two_reviewed_steps_are_valid(self):
        for step in (10000, 16000):
            quality.validate_step(step)
        for step in (-1, 0, 5500, 7000, 12000, 15999, 16001, 200000):
            with self.subTest(step=step), self.assertRaises((ValueError, RuntimeError)):
                quality.validate_step(step)

    def test_sha_is_exact_lowercase_hex(self):
        quality.validate_sha('0123456789abcdef' * 4)
        for sha in ('', 'a' * 63, 'a' * 65, 'A' * 64, 'g' * 64, '../' + 'a' * 61):
            with self.subTest(sha=sha), self.assertRaises((ValueError, RuntimeError)):
                quality.validate_sha(sha)

    def test_worker_guard_requires_real_uid_parent_root_and_window_identity(self):
        quality.guard_contract(*self.identities())
        for mutation in ('self_uid', 'parent_uid', 'ppid', 'module', 'root', 'anchor_pid', 'anchor_start'):
            me, parent, anchor, authorization, root = copy.deepcopy(self.identities())
            if mutation == 'self_uid':
                me['uid'] += 1
            elif mutation == 'parent_uid':
                parent['uid'] += 1
            elif mutation == 'ppid':
                me['ppid'] += 1
            elif mutation == 'module':
                parent['command'][2] = 'training.unrelated_worker'
            elif mutation == 'root':
                parent['command'][-1] += '-other'
            elif mutation == 'anchor_pid':
                anchor['child_pid'] += 1
            else:
                anchor['child_start'] = 'recycled-pid-start'
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                quality.guard_contract(me, parent, anchor, authorization, root)

    def test_worker_guard_requires_each_existing_authorization_field(self):
        for field in ('owner_authorized', 'exclusive_aegisvoice_v2_gpu', 'production_stopped', 'status'):
            for replacement in (None, False, 'inactive') if field == 'status' else (None, False):
                me, parent, anchor, authorization, root = self.identities()
                authorization[field] = replacement
                with self.subTest(field=field, replacement=replacement), self.assertRaises((ValueError, RuntimeError)):
                    quality.guard_contract(me, parent, anchor, authorization, root)

    def test_gpu_must_be_empty_before_load_and_exactly_our_pid_after_load(self):
        own_pid = 1235
        quality.gpu_ownership([], own_pid, loaded=False)
        quality.gpu_ownership([own_pid], own_pid, loaded=True)
        for pids, loaded in (([own_pid], False), ([999], False), ([], True), ([999], True), ([own_pid, 999], True)):
            with self.subTest(pids=pids, loaded=loaded), self.assertRaises((ValueError, RuntimeError)):
                quality.gpu_ownership(pids, own_pid, loaded=loaded)

    def test_finalization_preserves_failures_and_all_seven_denominators(self):
        cases = [dict(id=str(index)) for index in range(7)]
        source = [dict(case='0', outcome='ok'), dict(case='1', outcome='failed', error='synthetic failure')]
        original = copy.deepcopy(source)
        rows = quality.finalize_rows(source, cases, error='resource cap')
        self.assertEqual(source, original)
        self.assertEqual(len(rows), 7)
        by_case = {row['case']: row for row in rows}
        self.assertEqual(set(by_case), {case['id'] for case in cases})
        self.assertEqual(by_case['0']['outcome'], 'ok')
        self.assertEqual(by_case['1']['outcome'], 'failed')
        self.assertEqual(by_case['1']['error'], 'synthetic failure')
        self.assertTrue(all(by_case[str(index)]['outcome'] == 'not_attempted' for index in range(2, 7)))
        with self.assertRaises((ValueError, RuntimeError)):
            quality.validate_rows(rows, cases)

    def test_valid_synthetic_seven_wav_set_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            quality.validate_rows(rows, cases)

    def test_missing_duplicate_failed_or_unexpected_case_cannot_be_filtered_away(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            variants = [rows[:-1], rows[:-1] + [rows[0]],
                        [dict(row, outcome='failed') if index == 6 else row for index, row in enumerate(rows)],
                        [dict(row, case='unexpected') if index == 6 else row for index, row in enumerate(rows)]]
            for index, invalid in enumerate(variants):
                with self.subTest(variant=index), self.assertRaises((ValueError, RuntimeError)):
                    quality.validate_rows(invalid, cases)

    def test_recorded_wave_qc_requires_finite_bounded_unclipped_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            for field, value in (('finite', False), ('peak', 1.0001), ('peak', float('nan')), ('clipped_samples', 1)):
                invalid = copy.deepcopy(rows)
                invalid[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises((ValueError, RuntimeError)):
                    quality.validate_rows(invalid, cases)

    def test_wav_hash_must_match_the_complete_retained_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            rows[0]['wav_sha256'] = '0' * 64
            with self.assertRaises((ValueError, RuntimeError)):
                quality.validate_rows(rows, cases)

    def test_wav_contract_rejects_stereo_other_rate_and_other_sample_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            path = Path(rows[0]['wav'])
            for options in (dict(channels=2), dict(rate=22050), dict(width=3)):
                self.write_wave(path, **options)
                rows[0]['wav_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(options=options), self.assertRaises((ValueError, RuntimeError)):
                    quality.validate_rows(rows, cases)

    def test_wav_truncation_is_rejected_even_when_the_new_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, cases = self.synthetic_rows(Path(temporary))
            path = Path(rows[0]['wav'])
            path.write_bytes(path.read_bytes()[:-2])
            rows[0]['wav_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises((ValueError, RuntimeError)):
                quality.validate_rows(rows, cases)

    def test_budget_has_exact_working_disk_free_reserve_and_ram_boundaries(self):
        extra = 200_000_000
        used = 40_000_000_000 - extra
        free = 5 * 1024**3 + extra
        ram = 8 * 1024**3
        self.assertIsInstance(quality.budget_guard(used, free, extra, ram), dict)
        for values in ((used + 1, free, extra, ram), (used, free - 1, extra, ram), (used, free, extra, ram - 1)):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                quality.budget_guard(*values)

    def fake_runtime(self):
        dtype = object()

        def tensor(size):
            return SimpleNamespace(device=SimpleNamespace(type='cuda'), dtype=dtype, numel=lambda: size)

        def model(size):
            parameters, buffers = [tensor(size)], [tensor(1)]
            return SimpleNamespace(parameters=lambda: iter(parameters), buffers=lambda: iter(buffers),
                                   training=False, test_parameters=parameters, test_buffers=buffers)

        acoustic, vocoder = model(17), model(23)
        properties = SimpleNamespace(name='Synthetic GPU, no CUDA execution', total_memory=12 * 1024**3,
                                     major=8, minor=6)
        torch = SimpleNamespace(
            __version__='synthetic-torch', float32=dtype,
            version=SimpleNamespace(cuda='synthetic-cuda'),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=False, benchmark=True, deterministic=False, version=lambda: 999)),
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            get_float32_matmul_precision=lambda: 'synthetic-matmul-policy',
            cuda=SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: properties,
                                 is_bf16_supported=lambda: True, memory_allocated=lambda: 100,
                                 memory_reserved=lambda: 200, max_memory_allocated=lambda: 300,
                                 max_memory_reserved=lambda: 400))
        return torch, SimpleNamespace(acoustic=acoustic, vocoder=vocoder)

    def test_runtime_metadata_records_actual_flags_residency_and_memory(self):
        torch, backend = self.fake_runtime()
        metadata = quality.runtime_metadata(torch, backend)
        self.assertEqual(metadata['torch'], 'synthetic-torch')
        self.assertEqual(metadata['cuda'], 'synthetic-cuda')
        self.assertEqual(metadata['cudnn'], 999)
        self.assertEqual(metadata['device']['name'], 'Synthetic GPU, no CUDA execution')
        self.assertEqual(metadata['flags'], dict(cuda_matmul_allow_tf32=True, cudnn_allow_tf32=False,
                                               cudnn_benchmark=True, cudnn_deterministic=False,
                                               deterministic_algorithms=True, deterministic_warn_only=False,
                                               float32_matmul_precision='synthetic-matmul-policy'))
        self.assertEqual(metadata['precision'], dict(acoustic_autocast='bf16', parameters='FP32', vocoder='FP32'))
        self.assertEqual(metadata['vram'], dict(resident_allocated_bytes=100, resident_reserved_bytes=200,
                                              peak_allocated_bytes=300, peak_reserved_bytes=400))
        for name, count in (('acoustic', 17), ('vocoder', 23)):
            self.assertEqual(metadata['models'][name]['parameter_count'], count)
            self.assertTrue(metadata['models'][name]['all_parameters_cuda'])
            self.assertTrue(metadata['models'][name]['all_buffers_cuda'])
        json.dumps(metadata, allow_nan=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.assertFalse(quality.runtime_metadata(torch, backend)['flags']['cuda_matmul_allow_tf32'])
        self.assertTrue(metadata['flags']['cuda_matmul_allow_tf32'])

    def test_runtime_metadata_rejects_cpu_parameters_buffers_other_dtype_and_training(self):
        for condition in ('cpu_parameter', 'cpu_buffer', 'parameter_dtype', 'training'):
            torch, backend = self.fake_runtime()
            if condition == 'cpu_parameter':
                backend.acoustic.test_parameters[0].device.type = 'cpu'
            elif condition == 'cpu_buffer':
                backend.vocoder.test_buffers[0].device.type = 'cpu'
            elif condition == 'parameter_dtype':
                backend.vocoder.test_parameters[0].dtype = object()
            else:
                backend.acoustic.training = True
            with self.subTest(condition=condition), self.assertRaises(RuntimeError):
                quality.runtime_metadata(torch, backend)

    def test_quality_abort_escapes_measure_instead_of_becoming_a_case_failure(self):
        from training import v2_fastpitch_benchmark as common
        with tempfile.TemporaryDirectory() as temporary:
            backend = SimpleNamespace(
                np=SimpleNamespace(asarray=lambda value, **_: value, float32=object()),
                fp=SimpleNamespace(normalize=lambda value: value),
                torch=SimpleNamespace(manual_seed=lambda _: None),
                synchronize=lambda: None, reset_peak=lambda: None,
                generate=Mock(side_effect=quality.QualityAbort('synthetic resource abort')))
            case = dict(id='synthetic', text='synthetic only')
            plan = dict(expected_step=16000, effective_texts=dict(synthetic='synthetic only'))
            with patch.object(common, 'frontend', return_value=lambda text: dict(tts_text=text)):
                with self.assertRaises(quality.QualityAbort):
                    quality.measure(backend, case, plan, Path(temporary))
            backend.generate.assert_called_once()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_jobs_preserve_all_seven_cases_including_failed_and_not_attempted(self):
        cases = [dict(id=str(index), text='synthetic source ' + str(index)) for index in range(7)]
        rows = quality.finalize_rows([
            dict(case='0', outcome='ok', wav='/synthetic0.wav', wav_sha256='a' * 64),
            dict(case='1', outcome='failed', error='synthetic failure')], cases, error='resource cap')
        plan = dict(expected_step=16000, cases=cases,
                    spoken={case['id']: 'synthetic spoken ' + case['id'] for case in cases},
                    effective_texts={case['id']: 'synthetic effective ' + case['id'] for case in cases})
        jobs = quality.jobs_for(rows, plan)
        self.assertEqual(len(jobs), 7)
        self.assertEqual([job['case'] for job in jobs], [case['id'] for case in cases])
        self.assertEqual(jobs[0]['generation_outcome'], 'ok')
        self.assertTrue(jobs[0]['product_audio_valid'])
        self.assertEqual(jobs[0]['input_audio_sha256'], 'a' * 64)
        self.assertEqual(jobs[1]['generation_outcome'], 'failed')
        self.assertFalse(jobs[1]['product_audio_valid'])
        self.assertIsNone(jobs[1]['wav'])
        self.assertTrue(all(job['generation_outcome'] == 'not_attempted' for job in jobs[2:]))
        for job, case in zip(jobs, cases):
            self.assertEqual(job['expected_verbatim'], case['text'])
            self.assertEqual(job['expected_spoken'], plan['spoken'][case['id']])
            self.assertEqual(job['expected_effective'], plan['effective_texts'][case['id']])

    def test_module_import_and_help_do_not_import_torch_or_numpy(self):
        script = '''import runpy, sys
from training import v2_fastpitch_gpu_quality
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
sys.argv = ['v2_fastpitch_gpu_quality', '--help']
try:
    runpy.run_module('training.v2_fastpitch_gpu_quality', run_name='__main__')
except SystemExit as stopped:
    assert stopped.code == 0, stopped.code
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--plan', result.stdout)
        self.assertIn('--execute', result.stdout)

    def test_final_and_test50_flags_fail_in_parser_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for forbidden in ('--final', '--test50'):
                output = Path(temporary) / forbidden.removeprefix('--')
                command = [sys.executable, '-m', 'training.v2_fastpitch_gpu_quality', '--plan',
                           '--expected-step', '10000', '--checkpoint', '/nonexistent-synthetic-checkpoint',
                           '--checkpoint-sha256', 'a' * 64, '--output', str(output), forbidden]
                result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
                with self.subTest(forbidden=forbidden):
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('unrecognized arguments', result.stderr)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
