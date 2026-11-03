"""Real filesystem/receipt/continuation guards; no Torch, GPU, ASR or sleeps."""
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from training import v2_piper_quality_monitor as quality


class PiperMonitorTests(unittest.TestCase):
    def fixture(self, root, step=3, failure=None):
        run = root / 'run'; run.mkdir()
        source = run / 'probes' / (f'step-{step:08d}' if step else 'initial')
        source.mkdir(parents=True)
        probes = run / 'quality-monitor-piper' / 'probes'; probes.mkdir(parents=True)
        corpus = {f'case{i}': dict(text=f'Texto caso{i}.') for i in range(7)}
        spoken = {case: value['text'] for case, value in corpus.items()}
        state = run / 'last.pt'; state.write_bytes(b'full paired G+D fixture' + str(step).encode())
        digest = quality.sha(state)
        quality.write(run / 'last.pt.json', dict(step=step, sha256=digest, bytes=state.stat().st_size), run)
        rows = []; jobs = []
        for case, definition in corpus.items():
            row = dict(case=case, source_text=definition['text'], effective_text=definition['text'],
                       step=step, frontend='native', sampling=dict(noise_scale=0., noise_w=0., length_scale=1.),
                       outcome='failed' if failure == case else 'ok')
            if failure != case:
                wav = source / (case + '.wav')
                with wave.open(str(wav), 'wb') as stream:
                    stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
                    stream.writeframes(b'\x00\x04' * 1000)
                row.update(wav=str(wav), wav_sha256=quality.sha(wav))
                jobs.append(dict(case=case, wav=str(wav), expected_verbatim=definition['text'],
                                 expected_spoken=spoken[case], expected_effective=definition['text'],
                                 audio_sha256='producer collision must not be copied into scorer job'))
            rows.append(row)
        quality.write(source / 'results.json', rows, run)
        quality.write(source / 'asr-jobs.json', dict(jobs=jobs), run)
        pointer = run / 'last-probes.json'
        quality.write(pointer, dict(step=step, directory=str(source), checkpoint_sha256=digest), run)
        policy = dict(scorer=quality.SCORER_SHA, asr='a' * 64, versions=quality.VERSIONS,
                      trainer_recipe='b' * 64)
        job = quality.freeze(pointer, run, probes, corpus, spoken, policy)
        return run, source, probes, corpus, spoken, job

    def monitor(self, root, job):
        monitor = quality.Monitor.__new__(quality.Monitor)
        monitor.root = root; monitor.run = root / 'run'; monitor.out = monitor.run / 'quality-monitor-piper'
        monitor.probes = monitor.out / 'probes'; monitor.state = dict(jobs={job['key']: job}, best_asr=None)
        monitor.child = monitor.proc = None; monitor.running = True; monitor.policy = job['policy']
        monitor.log = logging.getLogger('piper-monitor-tests'); monitor.notices = []
        monitor.review = lambda kind, **values: monitor.notices.append((kind, values))
        monitor.policy_ok = lambda: True; monitor.budget_ok = lambda fresh=False: True
        monitor.a = SimpleNamespace(python='python3', asr_root=root / 'asr', timeout=1800)
        monitor.scorer = quality.REPO / 'benchmarks/v2/asr_score.py'
        monitor.last_poll = monitor.last_budget = monitor.last_heartbeat = 0.
        return monitor

    def attach_checkpoint(self, monitor, job):
        target = Path(job['directory']) / 'checkpoint.pt'
        job['checkpoint'] = quality.checkpoint(monitor.run, job['step'], job['checkpoint_sha256'], target)
        quality.write(target.parent / 'checkpoint.json', job['checkpoint'], monitor.probes)
        job['status'] = 'ready'

    def scores(self, job, errors=0):
        attempts = []
        for case, digest in job['audio_hashes'].items():
            texts = job['expected'][case]; hypothesis = texts['verbatim'] + (' erro' if errors else '')
            comparisons = {}
            for kind, text in texts.items():
                comparison = quality.plain_word_comparison(text, hypothesis)
                comparison.update(character_edits=5 if errors else 0, reference_characters=len(' '.join(quality.plain_word_tokens(text))))
                comparisons[kind] = comparison
            attempts.append(dict(candidate=job['candidate'], case=case, audio_sha256=digest,
                                 input_audio_sha256=digest, hypothesis=hypothesis, success=True,
                                 key=dict(candidate=job['candidate'], case=case, audio_sha256=digest, expected=texts),
                                 comparisons=comparisons))
        return dict(completed_utc='done', script_sha256=job['policy']['scorer'],
                    model_receipt_sha256=job['policy']['asr'], versions=quality.VERSIONS,
                    settings=dict(quality.SETTINGS, niceness=19), metric_policy=quality.METRIC_POLICY,
                    attempts=attempts)

    def test_module_import_has_no_tensor_or_asr_framework(self):
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'faster_whisper', 'ctranslate2') for name in sys.modules))

    def test_exact_seven_wavs_and_original_manifests_survive_source_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            run, source, probes, corpus, spoken, job = self.fixture(Path(temp))
            directory = Path(job['directory']); self.assertEqual(len(job['audio_hashes']), 7)
            self.assertEqual((source / 'case0.wav').stat().st_ino, (directory / 'case0.wav').stat().st_ino)
            for path in source.iterdir():
                path.unlink()
            source.rmdir()
            frozen = quality.read(directory / 'jobs.json', probes)['jobs']
            self.assertEqual(len(frozen), 7)
            self.assertTrue(all('audio_sha256' not in row and row['input_audio_sha256'] for row in frozen))
            self.assertEqual(quality.sha(directory / 'source-jobs.json'), job['source_jobs_sha256'])
            self.assertEqual(quality.sha(directory / 'pointer.json'), job['pointer_sha256'])
            self.assertNotIn('trainer_recipe', job['policy'])
            self.assertEqual(quality.read(directory / 'source.json', probes)['policy'], job['policy'])
            self.assertEqual(job['trainer_recipe_sha256'], 'b' * 64)

    def test_native_step_text_and_sampling_mismatches_are_rejected(self):
        for field, value in [('step', 4), ('source_text', 'Outro'), ('frontend', 'rewritten'), ('sampling', {'noise_scale': .5})]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                run, source, probes, corpus, spoken, job = self.fixture(Path(temp))
                rows = quality.read(source / 'results.json', run); rows[0][field] = value
                quality.write(source / 'results.json', rows, run)
                with self.assertRaises(quality.SafetyError):
                    quality.freeze(run / 'last-probes.json', run, probes, corpus, spoken, {})

    def test_pointer_path_escape_symlink_and_foreign_owner_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            pointer = quality.read(run / 'last-probes.json', run); pointer['directory'] = str(root / 'elsewhere')
            quality.write(run / 'last-probes.json', pointer, run)
            with self.assertRaises(quality.SafetyError):
                quality.freeze(run / 'last-probes.json', run, probes, corpus, spoken, {})
            alias = run / 'alias'; alias.symlink_to(source, target_is_directory=True)
            with self.assertRaises(quality.SafetyError):
                quality.owned_path(alias / 'case0.wav', run, 'file')
            with patch.object(quality.os, 'getuid', return_value=os.getuid() + 1):
                with self.assertRaises(quality.SafetyError):
                    quality.owned_path(source / 'case0.wav', run, 'file')

    def test_wav_hash_change_truncation_and_symlink_never_create_accepted_snapshot(self):
        for mode in ('hash', 'truncated', 'symlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
                wav = source / 'case0.wav'
                if mode == 'hash':
                    body = bytearray(wav.read_bytes()); body[-1] ^= 1; wav.write_bytes(body)
                elif mode == 'truncated':
                    wav.write_bytes(wav.read_bytes()[:-12])
                else:
                    original = source / 'original.wav'; wav.rename(original); wav.symlink_to(original)
                with self.assertRaises(quality.SafetyError):
                    quality.freeze(run / 'last-probes.json', run, probes, corpus, spoken, {'changed': mode})
                self.assertFalse(list(probes.glob('.snapshot-*')))

    def test_missing_synthesis_case_preserves_denominator_and_blocks_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root, failure='case6')
            monitor = self.monitor(root, job); monitor.capture()
            self.assertEqual(job['case_denominator'], 7); self.assertEqual(job['synthesis_successes'], 6)
            self.assertEqual(job['status'], 'generation_failed'); self.assertNotIn('checkpoint', job)
            self.assertFalse(quality.score_metrics(job, self.scores(job), 0)['comparable'])

    def test_duplicate_or_missing_probe_outcomes_not_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            run, source, probes, corpus, spoken, job = self.fixture(Path(temp))
            rows = quality.read(source / 'results.json', run); rows[-1] = rows[0]
            quality.write(source / 'results.json', rows, run)
            with self.assertRaises(quality.SafetyError):
                quality.freeze(run / 'last-probes.json', run, probes, corpus, spoken, {})

    def test_checkpoint_exact_best_used_after_last_advances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            os.link(run / 'last.pt', run / 'best.pt')
            quality.write(run / 'best.pt.json', quality.read(run / 'last.pt.json', run), run)
            new = run / 'new'; new.write_bytes(b'newer state'); new.replace(run / 'last.pt')
            quality.write(run / 'last.pt.json', dict(step=4, sha256=quality.sha(run / 'last.pt'), bytes=(run / 'last.pt').stat().st_size), run)
            target = Path(job['directory']) / 'checkpoint.pt'
            receipt = quality.checkpoint(run, 3, job['checkpoint_sha256'], target)
            self.assertEqual(receipt['source'], str(run / 'best.pt'))
            self.assertEqual(target.stat().st_ino, (run / 'best.pt').stat().st_ino)

    def test_wrong_step_or_sha_never_substitutes_newer_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            target = Path(job['directory']) / 'checkpoint.pt'
            self.assertIsNone(quality.checkpoint(run, 4, job['checkpoint_sha256'], target))
            self.assertIsNone(quality.checkpoint(run, 3, 'f' * 64, target)); self.assertFalse(target.exists())
            quality.write(run / 'last.pt.json', dict(step=3, sha256=job['checkpoint_sha256'], bytes=(run / 'last.pt').stat().st_size), run)
            (run / 'last.pt').write_bytes(b'wrong same published size'.ljust((run / 'last.pt').stat().st_size, b'.')[:(run / 'last.pt').stat().st_size])
            self.assertIsNone(quality.checkpoint(run, 3, job['checkpoint_sha256'], target)); self.assertFalse(target.exists())

    def test_atomic_checkpoint_race_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            target = Path(job['directory']) / 'checkpoint.pt'; real_link = os.link
            def racing_link(source, destination, **kwargs):
                replacement = run / 'replacement'; replacement.write_bytes(b'other step'); replacement.replace(source)
                real_link(source, destination, **kwargs)
            with patch.object(quality.os, 'link', side_effect=racing_link):
                self.assertIsNone(quality.checkpoint(run, 3, job['checkpoint_sha256'], target))
            self.assertFalse(target.exists())

    def test_orphan_exact_link_adopted_after_trainer_paths_disappear(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            target = Path(job['directory']) / 'checkpoint.pt'; os.link(run / 'last.pt', target)
            (run / 'last.pt').unlink(); (run / 'last.pt.json').unlink()
            binding = quality.checkpoint(run, 3, job['checkpoint_sha256'], target)
            self.assertEqual(binding['sha256'], job['checkpoint_sha256'])
            with patch.object(quality, 'CP_LIMIT', 1):
                with self.assertRaises(quality.SafetyError):
                    quality.checkpoint(run, 3, job['checkpoint_sha256'], target)

    def test_checkpoint_symlink_rejected_before_signal_or_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            target = Path(job['directory']) / 'checkpoint.pt'; target.symlink_to(run / 'last.pt')
            with self.assertRaises(quality.SafetyError):
                quality.checkpoint(run, 3, job['checkpoint_sha256'], target)

    def test_inventory_and_budget_count_unique_inodes_and_five_future_states(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            os.link(run / 'last.pt', run / 'best.pt'); os.link(run / 'last.pt', Path(job['directory']) / 'checkpoint.pt')
            inventory = quality.checkpoint_inventory(run, [job])
            self.assertEqual(inventory['unique_inodes'], 1)
            self.assertEqual(inventory['allocated_bytes'], (run / 'last.pt').stat().st_blocks * 512)
            self.assertEqual(quality.physical_usage([run, probes])['allocated_bytes'], quality.physical_usage([run])['allocated_bytes'])
            (run / 'last.pt.tmp').write_bytes(b'temporary')
            inventory2 = quality.checkpoint_inventory(run, [job]); self.assertEqual(inventory2['unique_inodes'], 2)
            usage = dict(allocated_bytes=30_000_000_000, permission_errors=[])
            plan = quality.budget_decision(usage, 30_000_000_000, inventory2)
            self.assertTrue(plan['ok']); self.assertEqual(plan['full_states'], 5)
            self.assertGreaterEqual(plan['reserve_bytes'], 5 * quality.BASE_BYTES - inventory2['allocated_bytes'])
            self.assertFalse(quality.budget_decision(dict(usage, allocated_bytes=39_000_000_000), 30_000_000_000, inventory2)['ok'])
            self.assertFalse(quality.budget_decision(usage, quality.FREE_RESERVE, inventory2)['ok'])
            self.assertFalse(quality.budget_decision(dict(usage, permission_errors=['unreadable']), 30_000_000_000, inventory2)['ok'])

    def test_pending_cap_counts_uncommitted_orphan_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); monitor.capture(); self.assertEqual(job['status'], 'ready')
            other = deepcopy(job); other.update(key='different', status='waiting_checkpoint')
            directory = probes / 'step-00000004-other'; directory.mkdir(); other['directory'] = str(directory)
            os.link(run / 'last.pt', directory / 'checkpoint.pt'); other.pop('checkpoint', None)
            monitor.state['jobs']['different'] = other
            with self.assertRaises(quality.SafetyError):
                monitor.capture()

    def test_capture_budget_blocks_before_link_and_missing_exact_state_reviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); monitor.state['budget'] = {'ok': False}; monitor.budget_ok = lambda fresh=False: False
            with patch.object(quality, 'checkpoint') as capture:
                monitor.capture(); capture.assert_not_called()
            monitor.budget_ok = lambda fresh=False: True
            quality.write(run / 'last.pt.json', dict(step=9, sha256='c' * 64, bytes=33), run)
            monitor.capture(); self.assertEqual(job['status'], 'checkpoint_unavailable')
            self.assertTrue(any(kind == 'exact_checkpoint_unavailable' for kind, _ in monitor.notices))

    def test_all_seven_complete_select_exact_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            directory = Path(job['directory']); quality.write(directory / 'scores.json', self.scores(job), probes)
            monitor.complete(dict(key=job['key']), 0)
            self.assertTrue(job['metrics']['comparable']); self.assertEqual(monitor.state['best_asr'], job['key'])
            before = len(monitor.notices); monitor.complete(dict(key=job['key']), None)
            self.assertEqual(len(monitor.notices), before)

    def test_partial_failed_missing_or_wrong_hash_scores_never_select_best(self):
        for mode in ('partial', 'failed', 'missing', 'hash', 'key', 'versions', 'settings', 'no_checkpoint'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
                monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job); score = self.scores(job)
                if mode == 'partial':
                    score['attempts'].pop()
                elif mode == 'failed':
                    score['attempts'][0]['success'] = False
                elif mode == 'hash':
                    score['attempts'][0]['audio_sha256'] = '0' * 64
                elif mode == 'key':
                    score['attempts'][0]['key']['candidate'] = 'different'
                elif mode == 'versions':
                    score['versions'] = dict(quality.VERSIONS, soundfile='other')
                elif mode == 'settings':
                    score['settings']['language'] = 'en'
                elif mode == 'no_checkpoint':
                    (Path(job['directory']) / 'checkpoint.pt').unlink()
                if mode != 'missing':
                    quality.write(Path(job['directory']) / 'scores.json', score, probes)
                monitor.complete(dict(key=job['key']), None)
                self.assertFalse(job['metrics']['comparable']); self.assertIsNone(monitor.state['best_asr'])
                self.assertEqual(job['metrics']['case_denominator'], 7)

    def test_best_requires_strictly_lower_wer_and_never_same_score_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            old = dict(key='earlier', metrics=dict(comparisons=dict(verbatim=dict(wer=0.))))
            monitor.state['jobs']['earlier'] = old; monitor.state['best_asr'] = 'earlier'
            quality.write(Path(job['directory']) / 'scores.json', self.scores(job), probes)
            monitor.complete(dict(key=job['key']), 0); self.assertEqual(monitor.state['best_asr'], 'earlier')

    def test_pause_pid_reuse_and_foreign_uid_never_signal(self):
        active = dict(pid=22, start_ticks='old', boot_id='boot', uid=os.getuid(), jobs='jobs', output='scores', asr_root='asr', paused=False)
        process = dict(pid=22, ppid=5, uid=os.getuid(), start_ticks='reused', command=[str(quality.REPO / 'benchmarks/v2/asr_score.py'), '--jobs', 'jobs', '--output', 'scores', '--asr-root', 'asr'])
        with patch.object(quality, 'process_identity', return_value=process), patch.object(quality, 'boot_id', return_value='boot'), patch.object(quality.os, 'getpgid', return_value=22), patch.object(quality.os, 'getsid', return_value=22), patch.object(quality.os, 'killpg') as kill:
            self.assertFalse(quality.owned_scorer(active))
            with self.assertRaises(quality.SafetyError):
                quality.pause_child(active, True)
            process['start_ticks'] = 'old'; process['uid'] = -1
            self.assertFalse(quality.owned_scorer(active)); kill.assert_not_called()

    def test_verified_pause_resume_and_shutdown_force_stop(self):
        active = dict(pid=22, paused=False)
        with patch.object(quality, 'owned_scorer', return_value=True), patch.object(quality, 'process_state', return_value=None), patch.object(quality.os, 'killpg') as kill:
            quality.pause_child(active, True); kill.assert_called_with(22, signal.SIGSTOP)
            quality.pause_child(active, False); kill.assert_called_with(22, signal.SIGCONT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); monitor.child = dict(pid=22, paused=True)
            with patch.object(quality, 'owned_scorer', return_value=True), patch.object(quality.os, 'killpg') as kill:
                monitor.shutdown(); kill.assert_called_once_with(22, signal.SIGSTOP)

    def test_pause_uses_actual_kernel_state_even_when_persisted_paused_flag_is_stale(self):
        active = dict(pid=22, paused=True)
        with patch.object(quality, 'owned_scorer', return_value=True), patch.object(quality, 'process_state', return_value='R'), patch.object(quality.os, 'killpg') as kill:
            quality.pause_child(active, True); kill.assert_called_once_with(22, signal.SIGSTOP)

    def test_dead_scorer_completion_waits_until_timing_pause_is_released(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); monitor.child = dict(pid=22, start_ticks='expired')
            (root / 'pause-asr').touch()
            with patch.object(quality, 'ticks', return_value=None), patch.object(monitor, 'complete') as complete:
                monitor.poll_child(); complete.assert_not_called()

    def test_pause_appearing_during_verification_prevents_popen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            monitor.verify_job = lambda job: (root / 'pause-asr').touch()
            with patch.object(quality.subprocess, 'Popen') as launch:
                monitor.start(); launch.assert_not_called()
            self.assertEqual(job['status'], 'ready'); self.assertFalse((monitor.out / 'launching.json').exists())

    def test_launch_is_cpu2_offline_nice19_identified_and_atomically_journaled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            process = SimpleNamespace(pid=123456789)
            with patch.object(quality.subprocess, 'Popen', return_value=process) as launch, patch.object(quality, 'ticks', return_value='launch-start'):
                monitor.start()
            command = launch.call_args.args[0]; env = launch.call_args.kwargs['env']
            self.assertEqual(command[:3], ['nice', '-n', '19'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['OMP_NUM_THREADS'], '2'); self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            active = quality.read(monitor.out / 'active.json', monitor.out)
            self.assertEqual(active['pid'], process.pid); self.assertEqual(active['start_ticks'], 'launch-start')
            self.assertEqual(active['key'], job['key']); self.assertEqual(job['status'], 'running')
            self.assertFalse((monitor.out / 'launching.json').exists())

    def test_missing_provider_process_handle_does_not_mark_live_scorer_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job)
            monitor.child = dict(key=job['key'], pid=123456789, start_ticks='live', paused=False,
                                 active_seconds=1., last_monotonic=quality.time.monotonic())
            with patch.object(quality, 'ticks', return_value='live'), patch.object(quality, 'owned_scorer', return_value=True), patch.object(quality, 'pause_child', return_value=False), patch.object(monitor, 'complete') as complete:
                monitor.poll_child(); complete.assert_not_called()
            self.assertIsNotNone(monitor.child)

    def test_old_lowercase_recipe_rejected_before_probe_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job)
            monitor.a.expected_manifest_sha256 = quality.TARGET_MANIFEST_SHA
            monitor.a.expected_fingerprint = quality.TARGET_FINGERPRINT
            recipe = dict(code_sha256=quality.TRAINER_SHA, frontend='native161', train_count=1025,
                          validation_count=55, data_manifest_sha256='0' * 64, data_fingerprint='1' * 64)
            quality.write(run / 'recipe.json', recipe, run)
            with patch.object(quality, 'freeze') as capture:
                with self.assertRaises(quality.SafetyError):
                    monitor.discover()
                capture.assert_not_called()

    def test_reconcile_live_launch_adopts_existing_scorer_without_popen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            intent = dict(key=job['key'], boot_id=quality.boot_id(), uid=os.getuid(), jobs='jobs', output='scores', asr_root='asr', paused=False)
            quality.write(monitor.out / 'launching.json', intent, monitor.out)
            current = os.getpid()
            process = dict(pid=current, ppid=1, start_ticks='existing-process', uid=os.getuid(), command=[])
            with patch.object(quality, 'process_identity', side_effect=lambda pid: process if pid == current else None), patch.object(quality, 'owned_scorer', side_effect=lambda active: active['pid'] == current), patch.object(quality.subprocess, 'Popen') as launch:
                monitor.reconcile_launch(); monitor.start(); launch.assert_not_called()
            self.assertEqual(monitor.child['pid'], current); self.assertEqual(job['status'], 'running')

    def test_reconcile_dead_launch_marks_partial_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job)
            score = self.scores(job); score['attempts'].pop(); score.pop('completed_utc')
            quality.write(Path(job['directory']) / 'scores.json', score, probes)
            intent = dict(key=job['key'], boot_id='prior-boot')
            quality.write(monitor.out / 'launching.json', intent, monitor.out)
            with patch.object(quality.subprocess, 'Popen') as launch:
                monitor.reconcile_launch(); monitor.start(); launch.assert_not_called()
            self.assertEqual(job['status'], 'failed'); self.assertIsNone(monitor.state['best_asr'])

    def trainer_fixture(self, root):
        (root / 'operations').mkdir()
        quality.write(root / 'operations/gpu-window.json', dict(child_pid=10, child_start='worker-start'), root)
        worker = dict(pid=20, worker_pid=10, status='running', started_utc='started', job=dict(command=['python', 'budget-gate.py']))
        processes = {10: dict(pid=10, ppid=1, start_ticks='worker-start', uid=os.getuid(), command=['python', '-m', 'training.v2_gpu_worker', '--root', str(root)]),
                     20: dict(pid=20, ppid=10, start_ticks='trainer-start', uid=os.getuid(), command=['python', '-m', 'training.v2_piper_experiment', 'pilot', '--run', str(root / 'run'), '--device', 'cuda', '--gpu-authorized'])}
        return worker, processes

    def test_trainer_binding_actual_child_uid_parent_start_and_worker_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); worker, processes = self.trainer_fixture(root)
            with patch.object(quality, 'process_identity', side_effect=lambda pid: deepcopy(processes.get(pid))):
                identity = quality.trainer_binding(root, root / 'run', worker)
                self.assertEqual(identity['pid'], 20); self.assertEqual(identity['job_command'], worker['job']['command'])
                processes[20]['start_ticks'] = 'reused'
                self.assertIsNone(quality.trainer_binding(root, root / 'run', worker, identity))
                processes[20]['start_ticks'] = 'trainer-start'; processes[20]['ppid'] = 30
                self.assertIsNone(quality.trainer_binding(root, root / 'run', worker))
                processes[20]['ppid'] = 10; processes[20]['uid'] = -1
                self.assertIsNone(quality.trainer_binding(root, root / 'run', worker))

    def test_retention_keeps_initial_latest_six_melbest_asrbest_and_active(self):
        jobs = [dict(key=str(i), step=i) for i in range(12)]
        selected = quality.retention_keys(jobs, '1', '3', 2)
        self.assertEqual(selected, {'0', '1', '2', '3', *map(str, range(6, 12))})

    def test_cleanup_drops_only_unselected_monitor_checkpoint_preserving_best_and_trainer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, source, probes, corpus, spoken, job = self.fixture(root)
            monitor = self.monitor(root, job); self.attach_checkpoint(monitor, job); job['status'] = 'complete'
            other = deepcopy(job); other.update(key='selected', step=4, directory=str(probes / 'step-00000004-selected'))
            directory = Path(other['directory']); directory.mkdir(); (directory / 'checkpoint.pt').write_bytes(b'selected')
            quality.write(directory / 'checkpoint.json', {'selected': True}, probes)
            monitor.state['jobs']['selected'] = other; monitor.state['best_asr'] = 'selected'
            original = (run / 'last.pt').read_bytes()
            with patch.object(quality.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'')):
                monitor.cleanup()
            self.assertFalse((Path(job['directory']) / 'checkpoint.pt').exists())
            self.assertTrue((directory / 'checkpoint.pt').exists())
            self.assertEqual((run / 'last.pt').read_bytes(), original)
            self.assertTrue((Path(job['directory']) / 'case0.wav').exists())

    def test_cleanup_requires_no_fuser_users_no_errors_owned_flat_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / 'old'; directory.mkdir(); (directory / 'file').write_text('kept')
            for code, stdout, stderr in [(0, b'pid', b''), (2, b'', b'unsupported')]:
                with patch.object(quality.subprocess, 'run', return_value=SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)):
                    self.assertFalse(quality.remove_directory(directory, root)); self.assertTrue((directory / 'file').exists())
            external = root / 'external'; external.write_text('preserve')
            (directory / 'link').symlink_to(external)
            with self.assertRaises(quality.SafetyError):
                quality.remove_directory(directory, root)
            (directory / 'link').unlink()
            with patch.object(quality.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'')):
                self.assertTrue(quality.remove_directory(directory, root))
            self.assertEqual(external.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
