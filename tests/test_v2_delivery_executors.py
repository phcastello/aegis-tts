"""Stdlib-only contract tests with synthetic text/PCM; no real final data or model."""
import copy
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from training import v2_delivery_protocol as protocol
from training import v2_http_delivery_benchmark as http
from training import v2_f5_delivery_benchmark as f5
from training import v2_voxcpm_delivery_launcher as launcher


class Sampler:
    def __init__(self,*_):self.peak=123.;self.error=None
    def __enter__(self):return self
    def __exit__(self,*_):pass


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.args=SimpleNamespace(final_test=False,decision=None,corpus=None,spoken=None,candidate='fixed',repetitions=3,
                                  run=self.root/'run',action='plan',gpu_authorized=False,delivery='sentence-files',
                                  endpoint='http://127.0.0.1:8002/v1/audio/speech',profile='expected',frontend='ptbr-numbers-v1',
                                  model_sha256='a'*64,startup_evidence=self.root/'startup.json',timeout=180.)
        self.cases=[dict(id=f'v_{i}',text=f'Frase sintética {i}.',size='short') for i in range(7)]
        self.cohort=dict(cases=self.cases,spoken={c['id']:c['text'] for c in self.cases},case_denominator=7,final_test=False)
        self.plan=dict(cohort=self.cohort,candidate='fixed',settings=dict(scope='synthetic'),source_hashes={},created_utc='fixed')

    def fake_corpus(self):
        here=self.root/'corpora';here.mkdir();(here/'validation.pt-BR.json').write_text(json.dumps(self.cases))
        (here/'spoken-validation.pt-BR.json').write_text(json.dumps(dict(cases=self.cohort['spoken'])))
        return here

    def valid_decision(self):
        value=dict(material_ttfa_gain_ms=400,max_wer_regression_pp=2,identity_rejection_rule='reject',
                   selected_candidate='winner',validation_evidence='synthetic',
                   comparison_roster=[dict(candidate='winner',role='finalist'),dict(candidate='fixed',role='baseline')])
        path=self.root/'decision.json';path.write_text(json.dumps(value));return path

    def test_final_guard_runs_before_any_corpus_read_or_hash(self):
        self.args.final_test=True;self.args.corpus=protocol.HERE/'test.pt-BR.json';self.args.spoken=self.root/'unread-spoken'
        for decision in (None,self.root/'missing.json'):
            self.args.decision=decision
            with patch.object(protocol,'read',side_effect=AssertionError('corpus opened')),patch.object(protocol,'sha',side_effect=AssertionError('corpus hashed')):
                with self.assertRaises((ValueError,FileNotFoundError)):protocol.load_cohort(self.args)
        self.args.decision=self.valid_decision();self.args.candidate='outside'
        with patch.object(protocol,'read',side_effect=AssertionError('corpus opened')),patch.object(protocol,'sha',side_effect=AssertionError('corpus hashed')):
            with self.assertRaisesRegex(ValueError,'outside'):protocol.load_cohort(self.args)

    def test_final_requires_explicit_path_and_spoken_after_global_roster_guard(self):
        self.args.final_test=True;self.args.decision=self.valid_decision()
        with patch.object(protocol,'read',side_effect=AssertionError('corpus opened')):
            with self.assertRaisesRegex(ValueError,'explicit canonical'):protocol.load_cohort(self.args)
            self.args.corpus=protocol.HERE/'test.pt-BR.json'
            with self.assertRaisesRegex(ValueError,'separately frozen'):protocol.load_cohort(self.args)
        here=self.fake_corpus();final_cases=self.cases[:6]
        (here/'test.pt-BR.json').write_text(json.dumps(final_cases));self.args.corpus=here/'test.pt-BR.json'
        self.args.spoken=self.root/'synthetic-spoken.json';self.args.spoken.write_text(json.dumps(dict(cases={c['id']:c['text'] for c in final_cases})))
        with patch.object(protocol,'HERE',here):value=protocol.load_cohort(self.args)
        self.assertEqual(value['decision_selected_candidate'],'winner');self.assertEqual(value['comparison_role'],'baseline');self.assertEqual(value['case_denominator'],6)

    def test_validation_has_no_alternate_corpus_and_requires_all_seven(self):
        self.args.corpus=self.root/'alternate'
        with patch.object(protocol,'read',side_effect=AssertionError('alternate opened')):
            with self.assertRaisesRegex(ValueError,'canonical validation7'):protocol.load_cohort(self.args)
        self.args.corpus=None;here=self.fake_corpus()
        with patch.object(protocol,'HERE',here):self.assertEqual(protocol.load_cohort(self.args)['case_denominator'],7)
        broken=self.cases[:-1]+[self.cases[0]];(here/'validation.pt-BR.json').write_text(json.dumps(broken))
        with patch.object(protocol,'HERE',here),self.assertRaisesRegex(ValueError,'duplicate'):protocol.load_cohort(self.args)

    def test_matrix_rejects_missing_or_duplicate_even_if_remaining_rows_successful(self):
        rows=[dict(case=c['id'],repetition=r,outcome='ok') for r in range(3) for c in self.cases]
        protocol.assert_matrix(rows,self.cohort,3)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaisesRegex(ValueError,'denominator'):protocol.assert_matrix(bad,self.cohort,3)

    def test_plan_is_frozen_before_execution_and_no_retry_over_partial(self):
        self.assertFalse(protocol.plan_or_verify(self.args,self.plan));self.args.action='execute'
        with self.assertRaisesRegex(ValueError,'authorization'):protocol.plan_or_verify(self.args,self.plan)
        self.args.gpu_authorized=True;new=dict(self.plan,created_utc='later');self.assertTrue(protocol.plan_or_verify(self.args,new));self.assertEqual(new['created_utc'],'fixed')
        bad=dict(self.plan,candidate='other')
        with self.assertRaisesRegex(ValueError,'differs'):protocol.plan_or_verify(self.args,bad)
        (self.args.run/'results.jsonl').write_text('partial')
        with self.assertRaisesRegex(ValueError,'Fresh'):protocol.plan_or_verify(self.args,self.plan)

    def startup_fixture(self):
        self.current=dict(pid=123,start_ticks='456',uid=os.getuid(),command=['python','-m','uvicorn'],boot_id='boot',start_utc_estimated='2026-11-04T00:00:00+00:00')
        value=dict(process=copy.deepcopy(self.current),boot_id='boot',spawn_utc='2026-11-04T00:00:00+00:00',
                   ready_utc='2026-11-04T00:03:00+00:00',spawn_to_ready_seconds=180.,internal_warmup=dict(count=3,description='actual three direct backend calls'),
                   endpoint=self.args.endpoint,candidate=self.args.candidate,model_sha256=self.args.model_sha256,seed_fixed=protocol.SEED,
                   runtime_source_hashes={'source.py':'b'*64},artifact_bindings=[dict(role='checkpoint',path='selected.pt',sha256=self.args.model_sha256)])
        return value

    def test_cold_receipt_requires_real_identity_clocks_seed_and_checkpoint(self):
        value=self.startup_fixture();http.startup_contract(self.args,value,self.current,'boot')
        changes=[('process',dict(self.current,start_ticks='reused')),('seed_fixed',protocol.SEED+1),('model_sha256','c'*64),
                 ('boot_id','oldboot'),('spawn_to_ready_seconds',float('nan')),('spawn_to_ready_seconds',3),
                 ('spawn_utc','2026-11-03T23:59:00+00:00'),('internal_warmup',dict(count=0,description='fake')),
                 ('artifact_bindings',[]),('runtime_source_hashes',{})]
        for key,replacement in changes:
            broken=copy.deepcopy(value);broken[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):http.startup_contract(self.args,broken,self.current,'boot')

    def test_http_completed_requires_status_and_exact_sse_named_events(self):
        row=dict(case='v_0',repetition=0,headers={'x-aegis-voice-profile':'expected','x-aegis-tts-frontend':'ptbr-numbers-v1','x-aegis-request-id':'id'})
        good=(200,{},b'{"request_id":"id","status":"completed"}')
        with patch.object(http.gateway,'complete_request',side_effect=[good,(200,{},b'event: accepted\n\nevent: completed\n\n')]):self.assertIn('sse',http.completed_contract(self.args,row))
        for response in [(200,{},b'{"request_id":"id","status":"failed"}'),(200,{},b'{"request_id":"other","status":"completed"}')]:
            with patch.object(http.gateway,'complete_request',return_value=response),self.assertRaises(ValueError):http.completed_contract(self.args,row)
        for events in (b'data: "event: completed"\n',b'event: accepted\n\nevent: failed\n\nevent: completed\n'):
            with patch.object(http.gateway,'complete_request',side_effect=[good,(200,{},events)]),self.assertRaises(ValueError):http.completed_contract(self.args,row)

    def test_failure_keeps_rep0_denominator_and_does_not_fabricate_effective_text(self):
        row=dict(candidate='fixed',case='v_0',repetition=0,final_test=False,outcome='failed')
        job=protocol.asr_job(row,self.cases[0],self.cases[0]['text'],self.root)
        self.assertEqual(job['generation_outcome'],'failed');self.assertNotIn('wav',job);self.assertNotIn('expected_effective',job)
        self.assertIsNone(protocol.asr_job(dict(row,repetition=1),self.cases[0],'text',self.root))

    def test_f5_oov_is_explicit_before_duration_or_ode_and_recorded_in_failure(self):
        backend=f5.Backend.__new__(f5.Backend)
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None))
        backend.np=SimpleNamespace();backend.args=self.args;backend.ref_text='a ';backend.ref_frames=1
        backend.chunks=lambda text:['synthetic-span'];backend.normalize=lambda text:dict(tts_text='aΩ')
        backend.tokens=lambda text:[['a','Ω']]
        backend.duration=Mock(side_effect=AssertionError('OOV reached duration'))
        backend.model=SimpleNamespace(vocab_char_map={' ':0,'a':1},sample=Mock(side_effect=AssertionError('OOV reached ODE')))
        with self.assertRaises(f5.OOVTokensError) as caught:backend.generate(self.cases[0],0)
        backend.duration.assert_not_called();backend.model.sample.assert_not_called()
        row=f5.failure_row(self.args,self.cases[0],0,caught.exception)
        self.assertEqual(row['oov_tokens'],['Ω']);self.assertEqual(row['failed_chunk_index'],0);self.assertEqual(row['outcome'],'failed')
        self.assertNotIn('wav',row);self.assertNotIn('ttfa_estimated_playback_ms',row)
        warm=f5.failure_row(self.args,dict(id='common_warmup'),0,caught.exception,True)
        self.assertEqual(warm['oov_tokens'],['Ω']);self.assertEqual(warm['condition'],'first-external-after-load')

    def test_f5_controlled_fake_backend_warms_three_then_keeps_failed_case_in_full_matrix(self):
        self.args.run.mkdir();protocol.save(self.args.run/'plan.json',self.plan);calls=[]
        def generate(case,rep,warmup=False):
            calls.append((case['id'],rep,warmup))
            if case['id']=='v_6' and rep==0:raise ValueError('synthetic generation failure')
            return dict(candidate='fixed',case=case['id'],size=case.get('size','warmup'),repetition=rep,condition='warm',final_test=False,seed=protocol.SEED,outcome='ok')
        fake=SimpleNamespace(metadata={},bindings={},generate=generate)
        m=protocol.metrics()
        with patch.object(protocol,'owner_guard',return_value={}),patch.object(protocol,'budget',return_value={}),patch.object(protocol,'gpu_set',return_value={}),patch.object(protocol,'audit_gpu_samples',return_value={}),patch.object(protocol,'metrics',return_value=m),patch.object(m,'GpuSampler',Sampler),patch.object(f5,'Backend',return_value=fake):
            f5.execute(self.args,self.plan)
        rows=[json.loads(x) for x in (self.args.run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls),24);self.assertEqual(calls[:3],[('common_warmup',i,True) for i in range(3)])
        self.assertEqual(len(rows),21);self.assertEqual(sum(r['outcome']=='failed' for r in rows),1)
        jobs=protocol.read(self.args.run/'asr-jobs.json');self.assertEqual(jobs['expected_conditions'],7);self.assertEqual(len(jobs['jobs']),7)
        self.assertEqual(jobs['jobs'][-1]['generation_outcome'],'failed')

    def test_http_controlled_client_has_three_warmups_fixed_payload_and_all_21_rows(self):
        self.args.run.mkdir();protocol.save(self.args.run/'plan.json',self.plan)
        value=self.startup_fixture();protocol.save(self.args.startup_evidence,value);calls=[];pcm=b'\0\x20'*2400
        def receive(endpoint,payload,timeout):
            calls.append(payload)
            return pcm,dict(outcome='ok',headers={},eof_ms=50.,audio_seconds=.1)
        def completed(args,row):
            if row['case']=='v_6' and row['repetition']==0:raise ValueError('synthetic failed SSE')
            return dict(status=dict(status='completed'),sse='synthetic')
        m=protocol.metrics()
        with patch.object(protocol,'owner_guard',return_value={}),patch.object(protocol,'budget',return_value={}),patch.object(protocol,'gpu_set',return_value={}),patch.object(protocol,'audit_gpu_samples',return_value={}),patch.object(protocol,'metrics',return_value=m),patch.object(m,'GpuSampler',Sampler),patch.object(m,'receive',side_effect=receive),patch.object(http,'verified_startup',return_value=value),patch.object(http,'request_count',side_effect=[0,24]),patch.object(http,'completed_contract',side_effect=completed),patch.object(http.gateway,'complete_request',return_value=(200,{'x-aegis-voice-profile':'expected'},b'')):
            http.execute(self.args,self.plan)
        rows=[json.loads(line) for line in (self.args.run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls),24);self.assertEqual([p['input'] for p in calls[:3]],[protocol.WARMUP]*3)
        self.assertTrue(all(p==dict(input=p['input'],stream=True,response_format='pcm') for p in calls))
        self.assertEqual(len(rows),21);self.assertTrue(all(r['seed']==protocol.SEED for r in rows))
        failed=next(r for r in rows if r['case']=='v_6' and r['repetition']==0)
        self.assertEqual(failed['outcome'],'failed_gateway_completion');self.assertTrue((self.args.run/failed['wav']).exists())
        self.assertEqual(len(protocol.read(self.args.run/'asr-jobs.json')['jobs']),7)
        warms=protocol.read(self.args.run/'warmups.json');self.assertEqual(warms[0]['condition'],'first-external-after-ready')

    def test_final_budget_failure_is_not_swallowed_by_either_executor(self):
        from contextlib import ExitStack
        m=protocol.metrics()
        for backend_name in ('f5','http'):
            self.args.run=self.root/backend_name;self.args.run.mkdir();protocol.save(self.args.run/'plan.json',self.plan)
            value=self.startup_fixture();protocol.save(self.args.startup_evidence,value)
            def generate(case,rep,warmup=False):
                return dict(candidate='fixed',case=case['id'],size=case.get('size','warmup'),repetition=rep,condition='warm',final_test=False,seed=protocol.SEED,outcome='ok')
            fake=SimpleNamespace(metadata={},bindings={},generate=generate)
            patches=[patch.object(protocol,'owner_guard',return_value={}),patch.object(protocol,'budget',side_effect=[{},RuntimeError('synthetic final budget cap')]),
                     patch.object(protocol,'gpu_set',return_value={}),patch.object(protocol,'audit_gpu_samples',return_value={}),
                     patch.object(protocol,'metrics',return_value=m),patch.object(m,'GpuSampler',Sampler),patch.object(f5,'Backend',return_value=fake),
                     patch.object(m,'receive',side_effect=lambda *_:(b'\0\x20'*2400,dict(outcome='ok',headers={},eof_ms=50.,audio_seconds=.1))),
                     patch.object(http,'verified_startup',return_value=value),patch.object(http,'request_count',side_effect=[0,24]),
                     patch.object(http,'completed_contract',return_value={}),patch.object(http.gateway,'complete_request',return_value=(200,{'x-aegis-voice-profile':'expected'},b''))]
            with self.subTest(backend=backend_name),ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                with self.assertRaisesRegex(RuntimeError,'Final budget guard failed'):
                    (f5.execute if backend_name=='f5' else http.execute)(self.args,self.plan)
            receipt=protocol.read(self.args.run/'execution.json')
            self.assertIn('synthetic final budget cap',receipt['error']);self.assertIn('synthetic final budget cap',receipt['final_working']['error'])
            self.assertEqual(receipt['observations'],21);self.assertEqual(receipt['successes'],21)

    def test_http_late_eof_is_failure_at_deadline_boundary_and_excluded_from_success_summary(self):
        base=dict(candidate='synthetic',case='v_fake',condition='warm',outcome='ok',eof_ms=50.,audio_seconds=.1,rtf=.5)
        exact=http.enforce_deadline(dict(base),.05);late=http.enforce_deadline(dict(base,eof_ms=110.,rtf=1.1),.05)
        self.assertEqual(exact['outcome'],'ok');self.assertEqual(late['outcome'],'failed_deadline')
        self.assertEqual(late['eof_ms'],110.);self.assertEqual(late['rtf'],1.1)
        summary=protocol.metrics().summarize([exact,late])['groups']['synthetic/warm/v_fake']
        self.assertEqual((summary['observations'],summary['successes'],summary['failures']),(2,1,1))
        self.assertEqual(summary['metrics']['eof_ms']['n'],1);self.assertEqual(summary['metrics']['eof_ms']['p50'],50.)

    def test_startup_verification_binds_live_config_environment_and_artifact_hash(self):
        value=self.startup_fixture();checkpoint=self.root/'selected.pt';checkpoint.write_bytes(b'synthetic-weight')
        self.args.model_sha256=protocol.sha(checkpoint);value['model_sha256']=self.args.model_sha256
        source=self.root/'runtime.py';source.write_text('synthetic source')
        config=self.root/'runtime.json';protocol.save(config,dict(adapter=str(checkpoint),adapter_sha256=self.args.model_sha256,frontend=self.args.frontend))
        value['runtime_source_hashes']={str(source):protocol.sha(source)}
        value['artifact_bindings']=[dict(role='checkpoint',path=str(checkpoint),sha256=self.args.model_sha256),dict(role='runtime_config',path=str(config),sha256=protocol.sha(config))]
        protocol.save(self.args.startup_evidence,value)
        with patch.object(protocol,'identity',return_value=self.current),patch.object(http.gateway,'RUNTIME_FILES',[]),patch.object(http,'live_config_path',return_value=config):
            self.assertEqual(http.verified_startup(self.args)['model_sha256'],self.args.model_sha256)
            with patch.object(http,'live_config_path',return_value=self.root/'different.json'),self.assertRaisesRegex(ValueError,'different runtime'):http.verified_startup(self.args)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):http.verified_startup(self.args)

    def test_gpu_overlap_check_and_pcm_complete_waveform_are_not_header_success(self):
        path=self.root/'gpu.jsonl';path.write_text('')
        with self.assertRaisesRegex(RuntimeError,'absent'):protocol.audit_gpu_samples(path,[123])
        path.write_text(json.dumps(dict(processes=[['123','1024']]))+'\n');self.assertEqual(protocol.audit_gpu_samples(path,[123])['samples'],1)
        path.write_text(json.dumps(dict(processes=[['123','1024'],['999','1']]))+'\n')
        with self.assertRaisesRegex(RuntimeError,'overlapped'):protocol.audit_gpu_samples(path,[123])
        wav=self.root/'synthetic.wav';protocol.write_pcm(wav,b'\x00\x20'*2400)
        self.assertEqual(http.gateway.wav_contract(wav.read_bytes())['pcm_bytes'],4800)
        with self.assertRaises(ValueError):http.gateway.wav_contract(wav.read_bytes()[:-2])

    def test_imports_and_default_plans_do_not_load_torch_numpy_or_model(self):
        command=[sys.executable,'-c',"import sys; from training import v2_delivery_protocol,v2_f5_delivery_benchmark,v2_http_delivery_benchmark,v2_voxcpm_delivery_launcher; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"]
        result=subprocess.run(command,cwd=protocol.REPO,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        for module,extra in [('training.v2_f5_delivery_benchmark',[]),('training.v2_http_delivery_benchmark',['--profile','synthetic','--model-sha256','a'*64,'--startup-evidence',str(self.root/'not-opened.json')])]:
            target=self.root/module.rsplit('.',1)[1]
            result=subprocess.run([sys.executable,'-m',module,'--run',str(target),'--candidate','synthetic',*extra],cwd=protocol.REPO,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);plan=protocol.read(target/'plan.json')
            self.assertFalse(plan['torch_imported']);self.assertFalse(plan['gpu_executed']);self.assertEqual(plan['seed_policy'],'fixed per call; never increment with repetition')
            self.assertEqual(plan['common_external_warmups']['count'],3);self.assertEqual(plan['cohort']['case_denominator'],7)

    def launcher_args(self):
        self.args.run.mkdir();protocol.save(self.args.run/'plan.json',self.plan)
        self.args.checkpoint=self.root/'selected.pt';self.args.checkpoint.write_bytes(b'synthetic-checkpoint')
        self.args.checkpoint_sha256=protocol.sha(self.args.checkpoint)
        protocol.save(self.args.checkpoint.with_suffix('.sha256.json'),dict(sha256=self.args.checkpoint_sha256))
        self.args.python=launcher.gateway.PYTHON;self.args.source=self.root/'source';self.args.model=self.root/'model'
        self.args.cache=self.root/'cache';self.args.overlay=self.root/'overlay';self.args.compiler_prerequisites=self.root/'compiler.json'
        self.args.port=8002;self.args.startup_ms=320.;self.args.ready_timeout=600.
        return launcher.decorate(self.args)

    def test_launcher_preserves_venv_and_plan_does_not_read_checkpoint_or_spawn(self):
        args=self.launcher_args();args.checkpoint=self.root/'nonexistent'
        with patch.object(subprocess,'Popen',side_effect=AssertionError('spawn from plan')):
            value=launcher.build_plan(args)
        self.assertFalse(value['torch_imported']);self.assertEqual(args.python,launcher.gateway.PYTHON)
        self.assertEqual(value['settings']['launcher']['startup_ms'],320.)
        self.assertIn('directly calls client.execute',value['settings']['launcher']['lifecycle'])

    def mock_launcher(self,fail_budget_after_contracts=False):
        args=self.launcher_args();seen=[]
        child=SimpleNamespace(pid=123456,stdout=io.BytesIO(b'synthetic startup\n'),returncode=None)
        child.poll=lambda:child.returncode
        child.wait=lambda timeout:setattr(child,'returncode',0) or 0
        child.send_signal=lambda sig:seen.append(('signal',sig))
        child.kill=lambda:seen.append(('kill',None))
        identity=dict(pid=child.pid,ppid=os.getpid(),start_ticks='123',uid=os.getuid(),boot_id='boot',command=launcher.child_command(args))
        def client_call(actual_args,plan):
            seen.append(('client_pid',os.getpid()));self.assertIs(actual_args,args)
            protocol.save(args.run/'execution.json',dict(observations=21,successes=21,error=None))
        def budget(*_):
            if fail_budget_after_contracts and ('contracts_done',True) in seen:raise RuntimeError('synthetic post-contract budget cap')
            return {}
        def contracts(*_):seen.append(('contracts_done',True));return dict(pass_synthetic=True)
        m=protocol.metrics()
        common_patches=[patch.object(protocol,'owner_guard',return_value={}),patch.object(protocol,'budget',side_effect=budget),
                        patch.object(protocol,'gpu_set',return_value={}),patch.object(protocol,'identity',return_value=identity),
                        patch.object(protocol,'audit_gpu_samples',return_value={}),patch.object(protocol,'metrics',return_value=m),
                        patch.object(m,'GpuSampler',Sampler),patch.object(m,'gpu_snapshot',return_value=dict(processes=[])),
                        patch.object(launcher.gateway,'verify_overlay',return_value={}),patch.object(launcher.gateway,'cpu_check',return_value={}),
                        patch.object(launcher,'artifact_bindings',return_value=[]),patch.object(launcher,'child_environment',return_value={}),
                        patch.object(launcher.gateway,'complete_request',return_value=(200,{'x-aegis-voice-profile':args.profile,'x-aegis-tts-dsp-variant':'passthrough'},b'')),
                        patch.object(launcher.gateway,'contracts',side_effect=contracts),
                        patch.object(launcher.socket,'socket'),patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0)),
                        patch.object(launcher.client,'execute',side_effect=client_call)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in common_patches:stack.enter_context(item)
            popen=stack.enter_context(patch.object(launcher.subprocess,'Popen',return_value=child))
            if fail_budget_after_contracts:
                with self.assertRaisesRegex(RuntimeError,'budget cleanup failed'):launcher.execute(args,self.plan)
            else:launcher.execute(args,self.plan)
        self.assertEqual(popen.call_count,1);self.assertEqual(popen.call_args.args[0],launcher.child_command(args))
        self.assertIn(('client_pid',os.getpid()),seen);self.assertTrue(any(item[0]=='signal' for item in seen))
        result=protocol.read(args.run/'launcher-result.json');self.assertFalse(result['server_left_running']);self.assertTrue(result['gpu_empty_after_cleanup'])
        self.assertIsNone(result['cleanup_error']);return result

    def test_launcher_calls_client_in_same_worker_pid_and_always_cleans_owned_server(self):
        self.assertIsNone(self.mock_launcher()['error'])

    def test_launcher_post_contract_budget_failure_invalidates_receipt_after_owned_cleanup(self):
        result=self.mock_launcher(True)
        self.assertIn('synthetic post-contract budget cap',result['error']);self.assertIn('synthetic post-contract budget cap',result['final_budget_error'])
        self.assertIn('synthetic post-contract budget cap',result['final_working']['error'])

    def test_owned_cleanup_rechecks_argv_before_term_and_after_timeout_before_kill(self):
        import signal
        identity=dict(pid=123,ppid=os.getpid(),start_ticks='1',uid=os.getuid(),boot_id='boot',command=['python','-m','uvicorn'])
        signals=[];child=SimpleNamespace(pid=123,returncode=None,poll=lambda:None,send_signal=signals.append,
                                       wait=lambda timeout:(_ for _ in ()).throw(subprocess.TimeoutExpired('synthetic',timeout)),kill=lambda:signals.append('KILL'))
        reused=dict(identity,command=['other-work'])
        with patch.object(protocol,'identity',return_value=reused),self.assertRaisesRegex(RuntimeError,'Refuse TERM'):launcher.stop_owned(child,identity)
        self.assertEqual(signals,[])
        with patch.object(protocol,'identity',side_effect=[identity,reused]),self.assertRaisesRegex(RuntimeError,'Refuse KILL'):launcher.stop_owned(child,identity)
        self.assertEqual(signals,[signal.SIGTERM])


if __name__=='__main__':unittest.main()
