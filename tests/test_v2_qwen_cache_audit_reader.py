"""Pinned CPU cache-reader proposal; all cache inputs are synthetic."""
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
from unittest.mock import patch
from training import v2_qwen_cache_audit_reader as reader


class CacheReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cache'; self.root.mkdir()
        self.file = self.root / 'synthetic-cache'; self.file.write_bytes(b'cache bytes')

    def frame(self):
        value = reader.snapshot(self.root, True)
        for row in value['entries'].values():
            row['metadata_signature'][3:5] = [reader.GOAL_START_NS - 1] * 2
            row['eligible_before_goal'] = True
        return value

    def test_default_cli_is_proposal_only_and_import_has_no_neural_dependencies(self):
        code = "import sys; from training import v2_qwen_cache_audit_reader; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
        self.assertEqual(subprocess.run([sys.executable, '-c', code], capture_output=True).returncode, 0)
        result = subprocess.run([sys.executable, '-m', 'training.v2_qwen_cache_audit_reader'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['actual_execution'])
        self.assertEqual(plan['status'], 'proposal_only_no_docker_operation')

    def test_cpu_proposal_pins_image_root_ro_gpu_void_and_exact_future_id(self):
        plan = reader.proposal('stat-stdio', 'aegis-qwen-cache-synthetic', Path(self.temp.name) / 'out')
        command = plan['create_argv']
        for flag, value in [('--runtime', 'runc'), ('--network', 'none'), ('--user', '0:0'), ('--cap-drop', 'ALL'), ('--pull', 'never'), ('--log-driver', 'none')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(command[command.index('--cap-add') + 1], 'DAC_READ_SEARCH')
        self.assertIn(reader.IMAGE, command); self.assertIn('--read-only', command)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void', command); self.assertIn('CUDA_VISIBLE_DEVICES=', command)
        mounts = [command[i+1] for i, value in enumerate(command) if value == '--mount']
        self.assertEqual(len(mounts), 2); self.assertTrue(all(value.endswith(',readonly') for value in mounts))
        self.assertNotIn('--gpus', command); self.assertNotIn('--privileged', command); self.assertNotIn('--device', command)
        self.assertEqual(plan['start_argv'][-1], '<CREATED_FULL_ID>')
        self.assertIn('-I', command); self.assertIn('-S', command); self.assertIn('-B', command)

    def test_goal_cutoff_requires_both_mtime_and_ctime_and_rejects_unknown(self):
        row = dict(metadata_signature=[1, 2, 3, reader.GOAL_START_NS, reader.GOAL_START_NS, 8, 1], uncertain_metadata=False)
        self.assertTrue(reader.eligible_before_goal(row))
        for index in (3, 4):
            invalid = copy.deepcopy(row); invalid['metadata_signature'][index] += 1
            self.assertFalse(reader.eligible_before_goal(invalid))
        self.assertFalse(reader.eligible_before_goal(dict(row, uncertain_metadata=True)))
        row['metadata_signature'][4] = None
        self.assertFalse(reader.eligible_before_goal(row))

    def test_quiet_hash_and_metadata_only_never_read_payload_in_stat_mode(self):
        value = reader.snapshot(self.root, True)
        self.assertTrue(value['complete'])
        self.assertEqual(value['entries'][self.file.name]['sha256'], hashlib.sha256(self.file.read_bytes()).hexdigest())
        with patch.object(reader.hashlib, 'sha256', side_effect=AssertionError('Payload hash in stat timing')):
            metadata = reader.snapshot(self.root, False)
        self.assertIsNone(metadata['entries'][self.file.name]['sha256'])
        self.assertEqual(metadata['physical_bytes'], value['physical_bytes'])

    def test_unchanged_post_goal_file_is_charged_even_with_identical_hash(self):
        baseline = self.frame()
        row = baseline['entries'][self.file.name]
        row['metadata_signature'][4] = reader.GOAL_START_NS + 1
        row['eligible_before_goal'] = False
        current = copy.deepcopy(baseline)
        fee = reader.accounted_fee(baseline, current)
        self.assertEqual(fee['accounted_bytes'], row['allocated_bytes'] + baseline['entries']['.']['allocated_bytes'])
        self.assertIn(self.file.name, fee['charged_paths'])
        self.assertTrue(fee['post_goal_files_always_charged'])

    def test_metadata_overwrite_of_original_file_loses_full_credit(self):
        baseline = self.frame(); current = copy.deepcopy(baseline)
        current['hashes_requested'] = False
        for row in current['entries'].values(): row['sha256'] = None
        directory_fee=baseline['entries']['.']['allocated_bytes']
        self.assertEqual(reader.accounted_fee(baseline, current)['accounted_bytes'], directory_fee)
        current['entries'][self.file.name]['metadata_signature'][4] += 1
        self.assertEqual(reader.accounted_fee(baseline, current)['accounted_bytes'], directory_fee+baseline['entries'][self.file.name]['allocated_bytes'])

    def test_permission_error_makes_whole_snapshot_incomplete_and_no_exclusion(self):
        denied=self.root/'denied';denied.mkdir()
        inode=denied.stat().st_ino; actual=os.listdir
        def listing(descriptor):
            if os.fstat(descriptor).st_ino==inode: raise PermissionError('synthetic denied')
            return actual(descriptor)
        with patch.object(reader.os,'listdir',side_effect=listing): value=reader.snapshot(self.root,True)
        self.assertFalse(value['complete']); self.assertEqual(value['eligible_before_goal_candidate_bytes'], 0)
        with self.assertRaisesRegex(ValueError, 'Complete'):
            reader.accounted_fee(value, value)

    def test_external_hardlink_stays_counted_even_if_temporally_original(self):
        os.link(self.file, self.root.parent / 'outside-scope')
        baseline = self.frame()
        fee = reader.accounted_fee(baseline, copy.deepcopy(baseline))
        self.assertEqual(fee['accounted_bytes'], (self.file.stat().st_blocks+self.root.stat().st_blocks)*512)

    def test_symlink_not_followed_and_entry_output_deadline_bounds_fail_closed(self):
        (self.root / 'external-symlink').symlink_to(self.root.parent)
        value = reader.snapshot(self.root, False)
        self.assertEqual(value['skipped_symlinks'], 1)
        self.assertFalse(value['complete']); self.assertEqual(value['eligible_before_goal_candidate_bytes'], 0)
        with patch.object(reader, 'MAX_ENTRIES', 1):
            self.assertFalse(reader.snapshot(self.root, False)['complete'])
        with patch.object(reader.time, 'monotonic', side_effect=[0, 181]), self.assertRaises(TimeoutError):
            reader.snapshot(self.root, False)
        with patch.object(reader, 'MAX_FRAME', 8), self.assertRaisesRegex(ValueError, 'frame'):
            reader.encode_frame(dict(large='not truncated'))

    def test_file_and_directory_symlinks_each_fail_whole_reader(self):
        for directory in (False, True):
            path = self.root / 'untrusted-link'
            path.symlink_to(self.root if directory else self.file)
            value = reader.snapshot(self.root, False)
            self.assertFalse(value['complete']); self.assertEqual(value['eligible_before_goal_candidate_bytes'], 0)
            with self.assertRaises(ValueError): reader.accounted_fee(value, value)
            path.unlink()

    def test_parent_directory_swap_never_hashes_outside_and_fails_roster(self):
        sub=self.root/'sub';sub.mkdir();(sub/'leaf').write_bytes(b'inside-only')
        outside=self.root.parent/'outside';outside.mkdir();(outside/'leaf').write_bytes(b'outside-forbidden')
        actual=os.open; swapped=False
        def opening(path,*args,**kwargs):
            nonlocal swapped
            if path=='leaf' and kwargs.get('dir_fd') is not None and not swapped:
                swapped=True;sub.rename(self.root/'kept-sub');sub.symlink_to(outside)
            return actual(path,*args,**kwargs)
        with patch.object(reader.os,'open',side_effect=opening): value=reader.snapshot(self.root,True)
        self.assertTrue(swapped);self.assertFalse(value['complete'])
        outside_digest=hashlib.sha256(b'outside-forbidden').hexdigest()
        self.assertTrue(all(row['sha256']!=outside_digest for row in value['entries'].values()))

    def test_stat_scan_created_subtree_is_detected_without_payload_hash(self):
        actual=os.stat; created=False
        def observed(path,*args,**kwargs):
            nonlocal created
            if path==self.file.name and kwargs.get('dir_fd') is not None and not created:
                created=True;(self.root/'new-subtree').mkdir();(self.root/'new-subtree'/'new-file').write_bytes(b'new')
            return actual(path,*args,**kwargs)
        with patch.object(reader.os,'stat',side_effect=observed),patch.object(reader.hashlib,'sha256',side_effect=AssertionError('stat payload hash')):
            value=reader.snapshot(self.root,False)
        self.assertTrue(created);self.assertFalse(value['complete']);self.assertEqual(value['eligible_before_goal_candidate_bytes'],0)

    def test_reader_guard_rejects_host_and_rw_mount_before_any_scan(self):
        with self.assertRaises(RuntimeError): reader.reader_guard(self.root)
        mounts = '42 1 0:1 / /input-cache rw,nosuid - ext4 /dev/synthetic rw\n'
        with patch.object(reader.os, 'geteuid', return_value=0), patch.object(Path, 'read_text', return_value=mounts):
            with self.assertRaisesRegex(RuntimeError, 'read-only'): reader.reader_guard(Path('/input-cache'))

    def test_persistent_reader_emits_ready_one_stat_per_request_then_quit(self):
        output = io.BytesIO()
        argv = ['reader', '--reader-mode', 'stat-stdio', '--root', str(self.root)]
        commands = [b'{"op":"snapshot","request_id":"synthetic-1"}', b'{"op":"quit"}']
        with patch.object(sys, 'argv', argv), patch.object(reader, 'command_lines', return_value=iter(commands)), patch.object(sys, 'stdout', SimpleNamespace(buffer=output)), patch.object(reader, 'reader_guard', return_value=dict(synthetic=True)), patch.object(reader.hashlib, 'sha256', side_effect=AssertionError('Hash in stat protocol')):
            reader.main()
        frames = [json.loads(row) for row in output.getvalue().splitlines()]
        self.assertEqual(len(frames), 3); self.assertTrue(frames[0]['ready'])
        self.assertEqual(frames[1]['request_id'], 'synthetic-1'); self.assertFalse(frames[1]['hashes_requested'])
        self.assertTrue(frames[2]['stopped'])

    def test_persistent_pipe_reader_is_bounded_and_does_not_lose_buffered_commands(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            os.write(write_fd, b'one\ntwo\n'); os.close(write_fd)
            self.assertEqual(list(reader.command_lines(stream)), [b'one', b'two'])
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            os.write(write_fd, b'x' * 4097); os.close(write_fd)
            with self.assertRaisesRegex(ValueError, 'bound'): list(reader.command_lines(stream))


if __name__ == '__main__': unittest.main()
