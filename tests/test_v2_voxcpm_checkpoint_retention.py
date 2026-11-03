"""Bounded exact validation-checkpoint retention; no tensors or live processes."""
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_voxcpm_quality_monitor as quality


class Retention3Tests(unittest.TestCase):
    def monitor(self,root):
        monitor=quality.Monitor.__new__(quality.Monitor)
        monitor.root=root;monitor.run=root/'run';monitor.out=monitor.run/'quality-monitor';monitor.probes=monitor.out/'probes'
        monitor.probes.mkdir(parents=True);monitor.state=dict(jobs={},best_asr='0',best_checkpoint='0');monitor.child=None
        monitor.log=logging.getLogger('retention3-tests');monitor.review=lambda *_,**__:None
        (monitor.run/'last.pt').write_bytes(b'trainer-last-protected')
        (monitor.run/'best.pt').write_bytes(b'trainer-best-protected')
        for step in (0,200,400,600,800):
            source=monitor.run/f'source-{step}.pt';source.write_bytes(f'weights-{step}'.encode())
            directory=monitor.probes/f'step-{step:08d}';directory.mkdir();os.link(source,directory/'checkpoint.pt')
            binding=dict(step=step,sha256=quality.sha(source));quality.write(directory/'checkpoint.sha256.json',binding)
            monitor.state['jobs'][str(step)]=dict(key=str(step),step=step,status='complete',directory=str(directory),checkpoint=binding,
                metrics=dict(comparable=True,comparisons=dict(verbatim=dict(wer=0. if step==0 else .5))))
        return monitor

    def test_latest_three_nonbest_exact_checkpoints_and_best_survive_safe_unlink(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=self.monitor(Path(temp));jobs=monitor.state['jobs'];old=Path(jobs['200']['directory'])
            with patch.object(quality.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as fuser:
                monitor.cleanup()
                self.assertTrue(any(str(old) in call.args[0] for call in fuser.call_args_list))
            for key in ('0','400','600','800'):
                path=Path(jobs[key]['directory']);self.assertTrue((path/'checkpoint.pt').exists());self.assertIn('checkpoint',jobs[key])
            self.assertFalse((old/'checkpoint.pt').exists());self.assertFalse((old/'checkpoint.sha256.json').exists());self.assertNotIn('checkpoint',jobs['200'])
            self.assertEqual((monitor.run/'source-200.pt').read_bytes(),b'weights-200')
            self.assertEqual((monitor.run/'last.pt').read_bytes(),b'trainer-last-protected')
            self.assertEqual((monitor.run/'best.pt').read_bytes(),b'trainer-best-protected')

    def test_in_use_old_checkpoint_is_preserved_until_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=self.monitor(Path(temp));old=Path(monitor.state['jobs']['200']['directory'])
            def fuser(command,**_):return SimpleNamespace(returncode=0 if str(old) in command else 1)
            with patch.object(quality.subprocess,'run',side_effect=fuser):monitor.cleanup()
            self.assertTrue((old/'checkpoint.pt').exists());self.assertIn('checkpoint',monitor.state['jobs']['200'])

    def test_cleanup_refuses_wrong_owner_or_outside_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);monitor=self.monitor(root)
            with patch.object(quality.os,'getuid',return_value=-1),patch.object(quality.subprocess,'run') as fuser:
                with self.assertRaisesRegex(ValueError,'Unsafe cleanup'):monitor.cleanup()
                fuser.assert_not_called()
            outside=root/'outside';outside.mkdir();(outside/'checkpoint.pt').write_bytes(b'preserve')
            monitor.state['jobs']={'200':dict(key='200',step=200,status='complete',directory=str(outside),checkpoint={})}
            with self.assertRaisesRegex(ValueError,'Unsafe cleanup'):monitor.cleanup()
            self.assertEqual((outside/'checkpoint.pt').read_bytes(),b'preserve')

    def test_completed_nonbest_step_can_bind_late_exact_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=self.monitor(Path(temp));directory=monitor.probes/'step-00001000';directory.mkdir()
            job=dict(key='1000',step=1000,status='complete',directory=str(directory));monitor.state['jobs']['1000']=job
            quality.write(monitor.run/'last.sha256.json',dict(step=800,sha256=quality.sha(monitor.run/'last.pt')))
            monitor.capture_checkpoints();self.assertNotIn('checkpoint',job)
            replacement=monitor.run/'new.pt';replacement.write_bytes(b'exact-1000');os.replace(replacement,monitor.run/'last.pt')
            quality.write(monitor.run/'last.sha256.json',dict(step=1000,sha256=quality.sha(monitor.run/'last.pt')))
            monitor.capture_checkpoints();self.assertEqual(job['checkpoint']['step'],1000)
            self.assertEqual(quality.sha(directory/'checkpoint.pt'),job['checkpoint']['sha256'])
            self.assertEqual((directory/'checkpoint.pt').stat().st_ino,(monitor.run/'last.pt').stat().st_ino)
            self.assertEqual(monitor.state['best_asr'],'0')

    def test_unbound_new_jobs_do_not_displace_last_three_exact_states(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=self.monitor(Path(temp));jobs=list(monitor.state['jobs'].values())
            jobs.extend(dict(key=str(step),step=step,status='pending') for step in (1000,1200))
            self.assertEqual(quality.checkpoint_retention(jobs,'0','0'),{'0','400','600','800'})
            self.assertEqual(quality.RESERVE,384*1024**2)


if __name__=='__main__':unittest.main()
