"""Failure evidence and crash recovery must not turn owner stops into retries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.v2_fastpitch_recovery_watch import (
    Watcher,classify,flags,invariant,next_resources,publish,read,retry_command,same_output,write,
)


class FastPitchRecoveryTests(unittest.TestCase):
    def test_only_final_confirmed_cuda_oom_can_retry(self):
        failed={'status':'failed','returncode':1}
        oom='torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.18 GiB.\n'
        self.assertEqual(classify(failed,oom),'cuda_oom')
        self.assertEqual(classify(failed,oom+'ValueError: NaN loss\n'),'other_error')
        self.assertEqual(classify(failed,'RuntimeError: CPU out of memory\n'),'other_error')
        self.assertEqual(classify({'status':'running'},oom),'other_error')
        self.assertEqual(classify(dict(failed,emergency='disk reserve'),oom),'other_error')
        for code in (-9,137,134):self.assertEqual(classify(dict(failed,returncode=code),oom),'other_error')

    def test_owner_sigint_and_clean_diagnostic_stop_never_restart(self):
        # Even an earlier OOM in a retained log cannot override the done receipt.
        tail='torch.OutOfMemoryError: CUDA out of memory.\n'
        for code in (0,130,143,-2,-15):
            self.assertEqual(classify({'status':'failed','returncode':code},tail),'normal_stop')

    def test_fallback_preserves_effective_batch_and_scientific_args(self):
        job={'command':['/usr/bin/env','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','python',
             'training/fastpitch_experiment.py','train','--output','/run','--precision','bf16',
             '--batch-size','8','--accumulation','2','--binary-start','1000','--resume']}
        self.assertEqual(next_resources(8,2),(4,4))
        self.assertEqual(next_resources(4,4),(2,8))
        self.assertEqual(next_resources(2,8),(1,16))
        self.assertIsNone(next_resources(1,16))
        with self.assertRaises(ValueError):next_resources(4,2)
        retry={'command':retry_command(job,4,4)}
        self.assertEqual(invariant(job),invariant(retry))
        self.assertEqual(flags(retry['command'])['--resume-allow-resource-change'],True)
        self.assertEqual(retry['command'].count('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'),1)

    def test_atomic_publication_rejects_overwrite_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'job.json';publish(path,{'command':['first']})
            publish(path,{'command':['first']})
            with self.assertRaises(ValueError):publish(path,{'command':['second']})
            self.assertEqual(read(path),{'command':['first']})

    def test_restart_reconciles_queue_publication_before_state_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'queue').mkdir();(root/'done').mkdir()
            job={'command':['wrapper'],'recovery':{'source_job':'073.json'}}
            watcher=Watcher.__new__(Watcher);watcher.root=root
            watcher.state={'jobs':{},'handled':{},'intents':{'key':{
                'status':'intent','job_name':'retry.json','job':job}}}
            write(root/'queue/retry.json',job)
            watcher.reconcile_intents()
            self.assertEqual(watcher.state['handled']['073.json']['action'],'queued_retry')
            self.assertEqual(watcher.state['intents']['key']['status'],'queued')
            # Repeated reconciliation and eventual completion preserve one lineage.
            write(root/'done/retry.json',{'job':job,'status':'succeeded','returncode':0})
            (root/'queue/retry.json').unlink();watcher.reconcile_intents()
            self.assertEqual(len(watcher.state['jobs']),1)
            self.assertEqual(watcher.state['intents']['key']['status'],'finished')

    def test_owner_recipe_change_and_short_completed_job_supersede_old_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'queue').mkdir();(root/'done').mkdir()
            watcher=Watcher.__new__(Watcher);watcher.root=root;watcher.run=root/'run'
            watcher.floor='2026-11-03T09:00:00+00:00';watcher.state={'jobs':{}}
            owner={'command':['python','fastpitch.py','train','--output',str(watcher.run),'--lr','.0001']}
            self.assertTrue(same_output(owner,watcher.run))
            completed={'job':owner,'started_utc':'2026-11-03T09:10:00+00:00','finished_utc':'2026-11-03T09:11:00+00:00'}
            write(root/'done/owner.json',completed);watcher.discover()
            self.assertEqual(watcher.newer_owner_training('failed.json',{'finished_utc':'2026-11-03T09:05:00+00:00'}),'owner.json')

    def test_filename_reuse_cannot_retry_the_cached_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'done').mkdir()
            watcher=Watcher.__new__(Watcher);watcher.root=root
            watcher.state={'handled':{}};notices=[]
            watcher.review=lambda kind,**kw:notices.append(kind);watcher.save=lambda:None
            cached={'command':['old recipe']}
            write(root/'done/reused.json',{'job':{'command':['new recipe']},'status':'failed','returncode':1})
            watcher.handle_done('reused.json',cached)
            self.assertEqual(watcher.state['handled']['reused.json']['action'],'review_only_job_identity_mismatch')
            self.assertEqual(notices,['job_filename_reused_or_receipt_changed'])

    def test_stop_during_cpu_audit_never_publishes_a_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('queue','done','logs','run/recovery-watch'):(root/name).mkdir(parents=True)
            watcher=Watcher.__new__(Watcher);watcher.root=root;watcher.run=root/'run'
            watcher.out=watcher.run/'recovery-watch';watcher.running=True
            job={'command':['python','fastpitch.py','train','--output',str(watcher.run),
                 '--batch-size','8','--accumulation','2','--precision','bf16']}
            watcher.baseline=invariant(job);watcher.state={'handled':{},'jobs':{'073.json':job},'intents':{}}
            watcher.review=lambda *a,**kw:None;watcher.save=lambda:None
            write(root/'done/073.json',dict(job=job,status='failed',returncode=1,
                  started_utc='2026-11-03T09:00:00+00:00',finished_utc='2026-11-03T09:05:00+00:00'))
            (root/'logs/073.log').write_text('torch.OutOfMemoryError: CUDA out of memory.\n')
            def interrupted_audit(*a,**kw):
                watcher.running=False
                # A valid-looking auditor result still cannot override stop.
                audit=Path(a[0][a[0].index('--output')+1]);write(audit,dict(status='pass',cuda_initialized=False))
                return SimpleNamespace(returncode=0)
            with patch('training.v2_fastpitch_recovery_watch.worker_ready',return_value=True),\
                 patch('training.v2_fastpitch_recovery_watch.gpu_processes',return_value=[]),\
                 patch('training.v2_fastpitch_recovery_watch.subprocess.run',side_effect=interrupted_audit),\
                 patch('training.v2_fastpitch_recovery_watch.publish') as publication:
                watcher.handle_done('073.json',job)
            publication.assert_not_called()
            self.assertEqual(list((root/'queue').iterdir()),[])
            self.assertEqual(watcher.state['intents'],{})


if __name__=='__main__':unittest.main()
