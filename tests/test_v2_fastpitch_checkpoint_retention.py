"""Only temporary files and stdlib; no Torch, GPU, trainer or real checkpoints."""
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_fastpitch_checkpoint_retention as retention


class RetentionTests(unittest.TestCase):
    def checkpoint(self,root,step=8000):
        root.mkdir(exist_ok=True);path=root/'last.pt';path.write_bytes(('weights-'+str(step)).encode())
        retention.write(root/'checkpoint.json',dict(step=step,checkpoint=str(path),sha256=retention.sha(path)))
        return path

    def packet(self,stage,step=8000):
        stage.mkdir();path=stage/'input-mels.pt';path.write_bytes(b'frozen-native-packet')
        return dict(step=step,sha256=retention.sha(path),bytes=path.stat().st_size)

    def test_late_publication_defers_wrong_step_then_binds_full_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';source=self.checkpoint(run,7999);stage=root/'stage';packet=self.packet(stage)
            self.assertIsNone(retention.bind_checkpoint(run,stage,packet));self.assertFalse((stage/'checkpoint.pt').exists())
            source=self.checkpoint(run,8000);record=retention.bind_checkpoint(run,stage,packet)
            self.assertEqual(record['step'],8000)
            self.assertEqual((stage/'checkpoint.pt').stat().st_ino,source.stat().st_ino)
            self.assertEqual(retention.sha(stage/'checkpoint.pt'),record['checkpoint_sha256'])

    def test_trainer_can_advance_after_link_without_invalidating_exact_old_inode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';source=self.checkpoint(run);stage=root/'stage';packet=self.packet(stage)
            digest=retention.sha
            def advance(path):
                result=digest(path)
                if Path(path).name=='checkpoint.pt':
                    replacement=run/'new.pt';replacement.write_bytes(b'weights-8001');os.replace(replacement,source)
                    retention.write(run/'checkpoint.json',dict(step=8001,checkpoint=str(source),sha256=digest(source)))
                return result
            with patch.object(retention,'sha',side_effect=advance):record=retention.bind_checkpoint(run,stage,packet)
            self.assertEqual(record['step'],8000)
            self.assertEqual((stage/'checkpoint.pt').read_bytes(),b'weights-8000')
            self.assertEqual(source.read_bytes(),b'weights-8001')

    def test_commit_recovers_after_link_before_manifest_even_when_source_is_gone(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';source=self.checkpoint(run);snapshots=root/'snapshots';snapshots.mkdir()
            stage=snapshots/'.stage-crashed';packet=self.packet(stage)
            retention.bind_checkpoint(run,stage,packet);source.unlink();(run/'checkpoint.json').unlink()
            record=retention.recover_stage(stage,snapshots);directory=Path(record['directory'])
            self.assertFalse(stage.exists());self.assertEqual(record['step'],8000)
            self.assertEqual(retention.sha(directory/'input-mels.pt'),packet['sha256'])
            self.assertEqual(retention.sha(directory/'checkpoint.pt'),record['checkpoint_sha256'])

    def test_budget_accounts_for_retention_plus_old_new_checkpoint_overlap(self):
        usage=dict(allocated_bytes=38_800_000_000,permission_errors=[])
        with patch.object(retention,'physical_usage',return_value=usage),patch.object(retention.shutil,'disk_usage',return_value=SimpleNamespace(free=100_000_000_000)):
            result=retention.budget(Path('/tmp'),550_000_000)
            self.assertTrue(result['allowed']);self.assertEqual(result['estimated_extra_bytes'],1_100_000_000+64*1024**2)
            usage['allocated_bytes']=39_000_000_000
            self.assertFalse(retention.budget(Path('/tmp'),550_000_000)['allowed'])
        usage['allocated_bytes']=1
        with patch.object(retention,'physical_usage',return_value=usage),patch.object(retention.shutil,'disk_usage',return_value=SimpleNamespace(free=5*1024**3+1_100_000_000+64*1024**2-1)):
            self.assertFalse(retention.budget(Path('/tmp'),550_000_000)['allowed'])

    def keeper(self,root):
        k=retention.Keeper.__new__(retention.Keeper);k.run=root/'trainer';k.run.mkdir();k.root=root;k.out=k.run/'asr-checkpoints';k.snapshots=k.out/'snapshots';k.snapshots.mkdir(parents=True)
        k.state=dict(snapshots={},best=None);k.last_heartbeat=0.;k.log=logging.getLogger('retention-tests');k.review=lambda *_,**__:None
        return k

    def test_pause_returns_before_any_heavy_iteration_work(self):
        with tempfile.TemporaryDirectory() as temp:
            k=self.keeper(Path(temp));(k.root/'pause-asr').touch()
            with patch.object(k,'reconcile_stages') as reconcile,patch.object(k,'scores') as scores,patch.object(k,'cleanup') as cleanup,patch.object(k,'capture') as capture,patch.object(retention,'sha') as sha,patch.object(retention,'physical_usage') as usage,patch.object(retention,'freeze_snapshot') as freeze,patch.object(retention,'in_use') as fuser:
                k.iteration()
                for heavy in (reconcile,scores,cleanup,capture,sha,usage,freeze,fuser):heavy.assert_not_called()
            self.assertTrue(retention.read(k.out/'status.json')['paused'])
            self.assertTrue(retention.read(k.out/'heartbeat.json')['paused'])
            self.assertFalse((k.out/'state.json').exists())

    def test_paused_constructor_defers_snapshot_discovery_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';run.mkdir();(root/'pause-asr').touch()
            args=SimpleNamespace(run=run,root=root,poll_seconds=2.)
            with patch.object(retention.Keeper,'reconcile_stages') as reconcile,patch.object(Path,'glob',side_effect=AssertionError('No filesystem crawl during pause')):
                k=retention.Keeper(args)
                reconcile.assert_not_called()
            for handler in list(k.log.handlers):handler.close();k.log.removeHandler(handler)

    def test_duplicate_stage_in_use_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';self.checkpoint(run);snapshots=root/'snapshots';snapshots.mkdir()
            stage=snapshots/'.stage-first';packet=self.packet(stage);record=retention.bind_checkpoint(run,stage,packet)
            retention.commit_stage(stage,snapshots,record)
            duplicate=snapshots/'.stage-second';packet=self.packet(duplicate);record=retention.bind_checkpoint(run,duplicate,packet)
            with patch.object(retention,'in_use',return_value=True) as fuser,patch.object(retention.shutil,'rmtree') as remove:
                with self.assertRaisesRegex(RuntimeError,'in use; preserved'):retention.commit_stage(duplicate,snapshots,record)
                fuser.assert_called_once_with(duplicate);remove.assert_not_called()
            self.assertTrue((duplicate/'checkpoint.pt').exists())

    def test_orphan_stage_in_use_is_preserved_during_recovery_and_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            k=self.keeper(Path(temp));stage=k.snapshots/'.stage-orphan';self.packet(stage)
            with patch.object(retention,'in_use',return_value=True) as fuser,patch.object(retention.shutil,'rmtree') as remove,patch.object(k,'review') as review:
                k.reconcile_stages()
                fuser.assert_called_once_with(stage);remove.assert_not_called();review.assert_called_once_with('staging_in_use',directory=str(stage))
            self.checkpoint(k.run);probe=k.run/'quality-monitor/probes/step-8000';probe.mkdir(parents=True)
            packet=probe/'input-mels.pt';packet.write_bytes(b'packet')
            job=dict(key='quality8000',step=8000,status='pending',directory=str(probe),snapshot=dict(step=8000,sha256=retention.sha(packet)))
            with patch.object(retention,'budget',return_value=dict(allowed=True)),patch.object(retention,'bind_checkpoint',return_value=None),patch.object(retention,'in_use',return_value=True) as fuser,patch.object(retention.shutil,'rmtree') as remove,patch.object(k,'review') as review:
                k.capture(dict(jobs={job['key']:job}))
                captured=k.snapshots/f'.stage-{os.getpid()}'
                fuser.assert_called_once_with(captured);remove.assert_not_called();review.assert_called_once_with('staging_in_use',directory=str(captured))
            self.assertTrue(captured.exists())

    def test_exact_quality_packet_capture_and_best_selection_never_use_later_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            k=self.keeper(Path(temp));source=self.checkpoint(k.run)
            probe=k.run/'quality-monitor/probes/step-8000';probe.mkdir(parents=True);packet=probe/'input-mels.pt';packet.write_bytes(b'packet')
            job=dict(key='quality8000',step=8000,status='pending',directory=str(probe),snapshot=dict(step=8000,sha256=retention.sha(packet)),policy_sha256='policy')
            quality=dict(jobs={job['key']:job},best_asr=None)
            with patch.object(retention,'budget',return_value=dict(allowed=True)):k.capture(quality)
            record=next(iter(k.state['snapshots'].values()));self.assertEqual(record['step'],8000)
            self.assertEqual(record['checkpoint_sha256'],retention.sha(source))
            job.update(status='complete',metrics=dict(source_sha256=job['snapshot']['sha256'],policy_sha256='policy',wer=.1))
            quality['best_asr']=job['key'];k.scores(quality)
            self.assertEqual(k.state['best'],record['key']);self.assertEqual(record['status'],'scored')
            # An obsolete scored link is not captured over and over at the same step.
            record['removed_utc']='removed'
            with patch.object(retention,'budget') as budget:k.capture(quality);budget.assert_not_called()

    def test_cleanup_removes_only_own_nonselected_links_and_keeps_one_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            k=self.keeper(Path(temp));last=self.checkpoint(k.run);best=k.run/'best.pt';best.write_bytes(b'trainer-best')
            for key,status,step in (('old','scored',6000),('best','scored',8000),('pending','pending_asr',10000)):
                directory=k.snapshots/key;directory.mkdir();os.link(last,directory/'checkpoint.pt')
                k.state['snapshots'][key]=dict(key=key,status=status,step=step,directory=str(directory),checkpoint_sha256=retention.sha(last))
            k.state['best']='best'
            with patch.object(retention,'in_use',return_value=False):k.cleanup()
            self.assertFalse((k.snapshots/'old').exists());self.assertTrue((k.snapshots/'best').exists());self.assertTrue((k.snapshots/'pending').exists())
            self.assertEqual(last.read_bytes(),b'weights-8000');self.assertEqual(best.read_bytes(),b'trainer-best')


if __name__=='__main__':unittest.main()
