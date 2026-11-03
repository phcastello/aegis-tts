"""Filesystem/race contracts only: no tensor library, model, GPU or ASR."""
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_voxcpm_quality_monitor as quality


class VoxQualityTests(unittest.TestCase):
    def trainer_fixture(self,root):
        worker=dict(pid=20,worker_pid=10,status='running',started_utc='2026-11-03T12:33:55+00:00',
                    job=dict(command=['python',str(root/'operations/advance-voxcpm-aegis.py')]))
        processes={10:dict(pid=10,ppid=1,start_ticks='worker-start',uid=os.getuid(),
                          command=['python','-m','training.v2_gpu_worker','--root',str(root)]),
                   20:dict(pid=20,ppid=10,start_ticks='trainer-start',uid=os.getuid(),
                          command=['python','-m','training.v2_voxcpm_train','--action','train','--run',str(root/'run')])}
        return worker,processes

    def test_gated_worker_command_binds_actual_exec_child_by_uid_parent_and_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);worker,processes=self.trainer_fixture(root)
            with patch.object(quality,'process_identity',side_effect=lambda pid:deepcopy(processes.get(pid))):
                binding=quality.trainer_binding(root,root/'run',worker)
                self.assertEqual(binding['pid'],20)
                self.assertEqual(binding['job_command'],worker['job']['command'])
                self.assertIn('training.v2_voxcpm_train',binding['command'])
                processes[20]['start_ticks']='reused-pid'
                self.assertIsNone(quality.trainer_binding(root,root/'run',worker,binding))
                processes[20]['start_ticks']='trainer-start';processes[20]['uid']=-1
                self.assertIsNone(quality.trainer_binding(root,root/'run',worker))
                processes[20]['uid']=os.getuid();processes[20]['ppid']=99
                self.assertIsNone(quality.trainer_binding(root,root/'run',worker))

    def test_observe_recognizes_gated_trainer_and_actual_progress_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir();out=run/'quality-monitor';out.mkdir()
            for name in ('queue','done'):(root/name).mkdir()
            worker,processes=self.trainer_fixture(root)
            quality.write(root/'worker-current.json',worker);quality.write(root/'queue/gated.json',worker['job'])
            quality.write(run/'progress.json',dict(step=17,loss=.5))
            monitor=quality.Monitor.__new__(quality.Monitor);monitor.root=root;monitor.run=run;monitor.out=out
            monitor.state={};monitor.child=None;monitor.last_budget=monitor.last_poll=monitor.last_heartbeat=0.
            monitor.log=logging.getLogger('gated-health-test');monitor.review=lambda *_,**__:None;monitor.budget_ok=lambda:True
            with patch.object(quality,'process_identity',side_effect=lambda pid:deepcopy(processes.get(pid))),\
                 patch.object(quality,'physical_usage',return_value=dict(allocated_bytes=0,permission_errors=[])):
                monitor.observe()
            status=quality.read(out/'status.json')
            self.assertTrue(status['trainer_running']);self.assertEqual(status['progress']['step'],17)
            self.assertEqual(status['progress_file'],str(run/'progress.json'))
            self.assertEqual(status['trainer_identity']['start_ticks'],'trainer-start')

    def test_checkpoint_uses_exact_best_when_last_has_advanced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=root/'snapshot.pt'
            for name,step in (('last',400),('best',200)):
                path=root/(name+'.pt');path.write_bytes(('weights-'+str(step)).encode())
                quality.write(root/(name+'.sha256.json'),dict(step=step,sha256=quality.sha(path)))
            binding=quality.checkpoint(root,200,target)
            self.assertEqual(binding['source'],str(root/'best.pt'))
            self.assertEqual(target.stat().st_ino,(root/'best.pt').stat().st_ino)

    def probes(self,root,failed=None):
        source=root/'native';source.mkdir();snapshots=root/'snapshots';snapshots.mkdir()
        corpus={str(i):dict(text='Texto '+str(i)) for i in range(7)}
        rows=[];jobs=[]
        for case,definition in corpus.items():
            row=dict(case=case,outcome='failed_PCMconversion' if case==failed else 'ok',nfe=10)
            if case!=failed:
                wav=source/(case+'.wav');wav.write_bytes(('audio-'+case).encode())
                row.update(wav=str(wav),sha256=quality.sha(wav))
                jobs.append(dict(case=case,wav=str(wav),expected_verbatim=definition['text']))
            rows.append(row)
        quality.write(source/'results.json',dict(step=200,results=rows))
        quality.write(source/'asr-jobs.json',dict(jobs=jobs))
        return source,snapshots,corpus

    def test_frozen_audio_survives_trainer_retention_and_binds_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            source,target,corpus=self.probes(Path(temp))
            job=quality.freeze(source/'results.json',target,corpus,{c:d['text'] for c,d in corpus.items()},{'scorer':'pinned'})
            frozen=Path(job['directory']);inode=(source/'0.wav').stat().st_ino
            self.assertEqual((frozen/'0.wav').stat().st_ino,inode)
            for path in source.iterdir():path.unlink()
            source.rmdir()
            self.assertEqual((frozen/'0.wav').read_bytes(),b'audio-0')
            self.assertEqual(quality.sha(frozen/'source-results.json'),job['source_results_sha256'])
            self.assertEqual(len(quality.read(frozen/'jobs.json')['jobs']),7)
            self.assertFalse(job['product'])

    def test_checkpoint_defers_old_or_racing_sidecar_and_recovers_orphan_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'trainer';run.mkdir();target=root/'checkpoint.pt'
            source=run/'last.pt';source.write_bytes(b'old-step')
            quality.write(run/'last.sha256.json',dict(step=100,sha256=quality.sha(source)))
            self.assertIsNone(quality.checkpoint(run,200,target));self.assertFalse(target.exists())
            replacement=run/'new';replacement.write_bytes(b'new-step');os.replace(replacement,source)
            quality.write(run/'last.sha256.json',dict(step=200,sha256=quality.sha(source)))
            os.link(source,target)  # Crash after hardlink, before sidecar/state.
            binding=quality.checkpoint(run,200,target)
            self.assertEqual(binding['step'],200)
            quality.write(target.with_suffix('.sha256.json'),binding)
            source.unlink();(run/'last.sha256.json').unlink()
            self.assertEqual(quality.checkpoint(run,200,target)['sha256'],quality.sha(target))
            target.unlink();source.write_bytes(b'incomplete-publication')
            quality.write(run/'last.sha256.json',dict(step=200,sha256='different'))
            self.assertIsNone(quality.checkpoint(run,200,target));self.assertFalse(target.exists())

    def monitor(self,root,job):
        monitor=quality.Monitor.__new__(quality.Monitor);monitor.run=root;monitor.root=root;monitor.out=root/'monitor';monitor.out.mkdir(exist_ok=True)
        monitor.probes=Path(job['directory']).parent;monitor.state=dict(jobs={job['key']:job},best_asr=None);monitor.child=None;monitor.running=True
        monitor.log=logging.getLogger('vox-quality-tests');monitor.review=lambda *_,**__:None
        return monitor

    def test_best_asr_can_capture_checkpoint_after_asr_finished_before_save(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'snapshot';directory.mkdir();job=dict(key='best',step=200,status='complete',directory=str(directory))
            monitor=self.monitor(root,job);monitor.state['best_asr']='best'
            monitor.capture_checkpoints();self.assertNotIn('checkpoint',job)
            source=root/'last.pt';source.write_bytes(b'complete-checkpoint')
            quality.write(root/'last.sha256.json',dict(step=200,sha256=quality.sha(source)))
            monitor.capture_checkpoints()
            self.assertEqual(job['checkpoint']['step'],200)
            self.assertEqual(monitor.state['best_checkpoint'],'best')
            self.assertEqual((directory/'checkpoint.pt').stat().st_ino,source.stat().st_ino)

    def test_seven_case_denominator_rejects_one_failed_asr_without_selecting_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,target,corpus=self.probes(root)
            policy=dict(scorer='scorer',asr='asr')
            job=quality.freeze(source/'results.json',target,corpus,{c:d['text'] for c,d in corpus.items()},policy)
            monitor=self.monitor(root,job);directory=Path(job['directory'])
            comparisons={k:dict(word_edits=0,reference_tokens=['texto']) for k in ('verbatim','spoken','effective')}
            attempts=[dict(case=c,audio_sha256=h,success=True,comparisons=comparisons) for c,h in job['audio_hashes'].items()]
            attempts[-1]['success']=False
            receipt=dict(completed_utc='done',script_sha256='scorer',model_receipt_sha256='asr',attempts=attempts)
            quality.write(directory/'scores.json',receipt);monitor.complete(dict(key=job['key']),0)
            self.assertEqual(job['metrics']['case_denominator'],7)
            self.assertEqual(job['metrics']['successful_cases'],6)
            self.assertFalse(job['metrics']['comparable']);self.assertEqual(job['metrics']['comparisons'],{})
            self.assertIsNone(monitor.state['best_asr'])
            job['status']='pending';attempts[-1]['success']=True;quality.write(directory/'scores.json',receipt)
            monitor.complete(dict(key=job['key']),0)
            self.assertTrue(job['metrics']['comparable']);self.assertEqual(monitor.state['best_asr'],job['key'])
            self.assertEqual(len((monitor.out/'metrics-history.jsonl').read_text().splitlines()),2)
            job['status']='running';monitor.complete(dict(key=job['key']),None)
            self.assertEqual(len((monitor.out/'metrics-history.jsonl').read_text().splitlines()),2)

    def test_retention_keeps_six_initial_best_training_best_and_active(self):
        jobs=[dict(key=str(i),step=i*200) for i in range(12)]
        self.assertEqual(quality.retention(jobs,'1',400,'3'),{'0','1','2','3',*map(str,range(6,12))})

    def test_pause_requires_verified_child_and_shutdown_forces_stop(self):
        active=dict(pid=123,paused=False)
        with patch.object(quality,'owned',return_value=False),patch.object(quality.os,'killpg') as kill:
            with self.assertRaises(RuntimeError):quality.pause_child(active,True)
            kill.assert_not_called()
        with patch.object(quality,'owned',return_value=True),patch.object(quality.os,'killpg') as kill:
            quality.pause_child(active,True);kill.assert_called_once_with(123,signal.SIGSTOP)
            kill.reset_mock();quality.pause_child(active,False);kill.assert_called_once_with(123,signal.SIGCONT)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);monitor=quality.Monitor.__new__(quality.Monitor);monitor.out=root;monitor.state={};monitor.child=dict(pid=123,paused=True)
            with patch.object(quality,'owned',return_value=True),patch.object(quality.os,'killpg') as kill:
                monitor.shutdown();kill.assert_called_once_with(123,signal.SIGSTOP)

    def test_pause_created_during_launch_checks_prevents_popen(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,target,corpus=self.probes(root)
            files={name:root/name for name in ('scorer','corpus','spoken')};asr=root/'asr';asr.mkdir();files['asr']=asr/'download-receipt.json'
            for path in files.values():path.write_bytes(b'pinned')
            policy={name:quality.sha(path) for name,path in files.items()}
            job=quality.freeze(source/'results.json',target,corpus,{c:d['text'] for c,d in corpus.items()},policy)
            monitor=self.monitor(root,job);monitor.a=SimpleNamespace(python='python3',asr_root=asr);monitor.policy=policy
            monitor.scorer=files['scorer'];monitor.corpus=files['corpus'];monitor.spoken=files['spoken'];monitor.budget_ok=lambda:True
            real_sha=quality.sha
            def digest(path):
                if Path(path).name=='6.wav':(root/'pause-asr').touch()
                return real_sha(path)
            with patch.object(quality,'sha',side_effect=digest),patch.object(quality.subprocess,'Popen') as launch:
                monitor.start();launch.assert_not_called()

if __name__=='__main__':unittest.main()
