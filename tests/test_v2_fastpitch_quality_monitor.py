import json
import logging
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training import v2_fastpitch_quality_monitor as quality


class QualityMonitorTests(unittest.TestCase):
    def prepared(self, root, step=4000):
        metadata = root/'validation-latest.json'
        packet = root/'validation-mels.pt'
        metadata.write_text(json.dumps(dict(step=step)))
        packet.write_bytes(b'complete-old-native-packet')
        # Explicit nanosecond order models the trainer's actual publication.
        os.utime(metadata, ns=(10_000_000, 10_000_000))
        os.utime(packet, ns=(20_000_000, 20_000_000))
        return metadata, packet

    def test_atomic_replacement_cannot_change_a_frozen_packet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, packet = self.prepared(root)
            frozen = root/'snapshot.pt'
            receipt = quality.freeze_snapshot(root, frozen)
            new = root/'new.pt'; new.write_bytes(b'new-packet')
            os.replace(new, packet)
            self.assertEqual(frozen.read_bytes(), b'complete-old-native-packet')
            self.assertNotEqual(frozen.stat().st_ino, packet.stat().st_ino)
            self.assertEqual(receipt['sha256'], quality.sha(frozen))
            self.assertEqual(receipt['step'], 4000)

    def test_metadata_published_before_its_packet_defers_the_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata, _ = self.prepared(root)
            os.utime(metadata, ns=(30_000_000, 30_000_000))
            with self.assertRaises(quality.SnapshotChanged):
                quality.freeze_snapshot(root, root/'snapshot.pt')
            self.assertFalse((root/'snapshot.pt').exists())

    def test_replacement_during_link_is_rejected_without_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata, packet = self.prepared(root)
            real_link = os.link
            def racing_link(source, target):
                replacement = root/'new.pt'; replacement.write_bytes(b'other-step')
                os.replace(replacement, packet)
                metadata.write_text(json.dumps(dict(step=4500)))
                real_link(source, target)
            with patch.object(quality.os, 'link', side_effect=racing_link):
                with self.assertRaises(quality.SnapshotChanged):
                    quality.freeze_snapshot(root, root/'snapshot.pt')
            self.assertFalse((root/'snapshot.pt').exists())

    def test_latest_six_best_and_active_are_preserved(self):
        jobs = [dict(key=str(i), step=i*2000) for i in range(12)]
        self.assertEqual(quality.retention_keys(jobs, '1', '2'), {'1', '2', *map(str, range(6,12))})

    def test_pause_signals_only_a_verified_owned_group(self):
        active = dict(pid=123, paused=False)
        with patch.object(quality, 'owned_group', return_value=False), patch.object(quality.os, 'killpg') as kill:
            with self.assertRaises(RuntimeError):
                quality.set_paused(active, True)
            kill.assert_not_called()
        with patch.object(quality, 'owned_group', return_value=True), patch.object(quality.os, 'killpg') as kill:
            self.assertTrue(quality.set_paused(active, True))
            kill.assert_called_once_with(123, signal.SIGSTOP)
            self.assertTrue(active['paused'])
            kill.reset_mock()
            self.assertTrue(quality.set_paused(active, False))
            kill.assert_called_once_with(123, signal.SIGCONT)

    def test_pause_created_during_launch_preparation_prevents_spawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probes = root/'probes'; probes.mkdir()
            directory = probes/'step-4000'; directory.mkdir()
            snapshot = directory/'input-mels.pt'; snapshot.write_bytes(b'packet')
            monitor = quality.Monitor.__new__(quality.Monitor)
            monitor.running=True; monitor.child=None; monitor.root=root; monitor.run=root; monitor.probes=probes
            monitor.decoder=root/'decoder'; monitor.scorer=root/'scorer'
            monitor.a=SimpleNamespace(inputs=root/'inputs', python='python3', vocoder=root/'vocoder')
            monitor.policy=dict(decoder_sha256='decoder', scorer_sha256='scorer', validation_manifest_sha256='inputs')
            monitor.policy_sha='policy'
            monitor.state=dict(jobs={'key':dict(status='pending', step=4000, directory=str(directory),
                                snapshot=dict(sha256='snapshot'), policy_sha256='policy')})
            monitor.budget_ok=lambda:True
            def digest(path):
                if path==snapshot:
                    (root/'pause-asr').touch()
                    return 'snapshot'
                return {monitor.decoder:'decoder', monitor.scorer:'scorer',
                        monitor.a.inputs/'validation.jsonl':'inputs'}[path]
            with patch.object(quality, 'sha', side_effect=digest), patch.object(quality.subprocess, 'Popen') as launch:
                monitor.start()
                launch.assert_not_called()

    def test_shutdown_pauses_a_child_even_if_cached_flag_already_says_paused(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=quality.Monitor.__new__(quality.Monitor)
            monitor.out=Path(temp); monitor.child=dict(pid=123, paused=True)
            monitor.state=dict(jobs={})
            with patch.object(quality, 'owned_group', return_value=True), patch.object(quality.os, 'killpg') as kill:
                monitor.pause_for_shutdown()
                kill.assert_called_once_with(123, signal.SIGSTOP)
                self.assertTrue(quality.read(monitor.out/'active.json')['paused'])

    def test_already_complete_probe_is_not_counted_twice_after_crash(self):
        monitor=quality.Monitor.__new__(quality.Monitor)
        monitor.state=dict(jobs={'key':dict(status='complete')})
        with patch.object(quality, 'read') as read:
            monitor.complete(dict(key='key'), None)
            read.assert_not_called()


if __name__=='__main__':
    unittest.main()
