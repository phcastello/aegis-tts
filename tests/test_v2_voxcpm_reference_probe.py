"""CPU-only boundary, API dispatch and cancellation tests; no TTS model."""
from contextlib import contextmanager
from types import SimpleNamespace
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from training import v2_voxcpm_reference_probe as probe


class Feature:
    shape=(64,4,64)
    dtype='torch.float32'
    device=SimpleNamespace(type='cpu')


def reference():return {'mode':'reference','ref_audio_feat':Feature()}


class ClosingStream:
    def __init__(self,items):self.items=iter(items);self.closed=0
    def __iter__(self):return self
    def __next__(self):return next(self.items)
    def close(self):self.closed+=1


class Model:
    def __init__(self,items=None):
        self.calls=[];self.stream=None
        self.items=items if items is not None else [(np.full(96,.05,dtype=np.float32),None,None)]*3
    def text_tokenizer(self,text):return list(text)
    def _generate_with_prompt_cache(self,**kwargs):
        self.calls.append(kwargs);self.stream=ClosingStream(self.items);return self.stream
    def generate_with_prompt_cache_streaming(self,**kwargs):return self._generate_with_prompt_cache(**kwargs)
    def _generate(self):return 'original'


class ReferenceControlTests(unittest.TestCase):
    def test_fixed_equivalence_excludes_cold_mismatch_and_requires_exact_last_pair(self):
        calls=[];saved=[]
        def perform(case,api,rep):
            calls.append((case,api,rep))
            fingerprint='cold-first' if rep==0 and api=='legacy' else 'fixed-warm'
            row=dict(outcome='ok',pcm_sha256=fingerprint,native_float32_sha256=fingerprint,native_samples=100,audio_seconds=1.)
            return b'',row,None
        result=probe.fixed_api_equivalence(perform,lambda c,r,p,e:saved.append((c,r,e)))
        self.assertEqual(len(calls),18);self.assertEqual(len(saved),9)
        self.assertTrue(all(r['exact_equal'] for r in result.values()))
        self.assertTrue(all(not r['rounds'][0]['exact_equal'] and r['rounds'][2]['exact_equal'] for r in result.values()))
        self.assertEqual(calls[:6],[(probe.CASES[0],a,r) for r in range(3) for a in (('legacy','cached') if r%2==0 else ('cached','legacy'))])

    def test_persistent_api_difference_never_retries_until_match(self):
        calls=[]
        def perform(case,api,rep):
            calls.append((case,api,rep))
            return b'',dict(outcome='ok',pcm_sha256=api,native_float32_sha256=api,native_samples=100,audio_seconds=1.),None
        result=probe.fixed_api_equivalence(perform,lambda *a:None)
        self.assertEqual(len(calls),18)
        self.assertTrue(all(not r['exact_equal'] for r in result.values()))

    def test_failed_initial_call_is_not_hidden_by_equal_last_pair(self):
        def perform(case,api,rep):
            return b'',dict(outcome='failed' if rep==0 and api=='legacy' else 'ok',pcm_sha256='same',native_float32_sha256='same',native_samples=100,audio_seconds=1.),None
        result=probe.fixed_api_equivalence(perform,lambda *a:None)
        self.assertTrue(all(not r['exact_equal'] and not r['all_calls_ok'] for r in result.values()))

    def test_utf8_io_under_real_ascii_locale_and_compact_unicode_error(self):
        code=r'''
import json,locale,sys,tempfile
from pathlib import Path
from training import v2_voxcpm_reference_probe as p
assert sys.flags.utf8_mode==0
locale.setlocale(locale.LC_CTYPE,'C')
assert locale.getencoding().lower() in ('ansi_x3.4-1968','ascii','us-ascii')
with tempfile.TemporaryDirectory() as d:
 f=Path(d)/'utf8.json';value={'text':'a\u00e7\u00e3o'}
 f.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
 try:f.read_text()
 except UnicodeDecodeError:pass
 else:raise AssertionError('Control default read unexpectedly accepted UTF8')
 assert p.read_json(f)==value
 p.pilot.atomic_json(f,value)
 assert json.loads(f.read_bytes().decode('utf-8'))==value
e=UnicodeDecodeError('ascii',b'\xff'+b'PRIVATE_TRANSCRIPT'*200000,0,1,'invalid')
msg=p.compact_error(e)
assert len(msg)<=512 and 'PRIVATE_TRANSCRIPT' not in msg
assert msg.startswith('UnicodeDecodeError:')
assert len(p.compact_error(RuntimeError('x'*2000000)))==512
print('ascii-locale UTF8 IO and bounded errors passed')
'''
        env=dict(os.environ,PYTHONUTF8='0',LC_ALL='C',CUDA_VISIBLE_DEVICES='')
        completed=subprocess.run([sys.executable,'-c',code],cwd=probe.REPO,env=env,capture_output=True,timeout=30)
        self.assertEqual(completed.returncode,0,(completed.stdout+completed.stderr).decode('utf-8',errors='replace')[:2000])

    def test_exact_boundary_covers_both_reference_markers_and_full400(self):
        self.assertEqual(probe.guard(556,reference())['reserved_slots'],1024)
        self.assertEqual(probe.guard(622,None)['reserved_slots'],1024)
        self.assertEqual(probe.guard(145,reference())['prefix_slots'],212)
        for tokens,cache in [(557,reference()),(623,None)]:
            with self.assertRaisesRegex(ValueError,'refuse truncation'):probe.guard(tokens,cache)
        for tokens in [True,-1,1.5]:
            with self.assertRaises(ValueError):probe.guard(tokens,None)

    def test_continuation_unknown_keys_wrong_shape_dtype_and_gpu_staging_rejected(self):
        variants=[{'mode':'continuation','audio_feat':Feature()},dict(reference(),prompt_text='caption')]
        for attr,value in [('shape',(63,4,64)),('dtype','torch.bfloat16'),('device',SimpleNamespace(type='cuda'))]:
            feat=Feature();setattr(feat,attr,value);variants.append({'mode':'reference','ref_audio_feat':feat})
        for cache in variants:
            with self.assertRaises(ValueError):probe.guard(3,cache)

    def test_cache_dispatch_uses_none_and_reference_and_rejects_before_native_call(self):
        model=Model();state={}
        with probe.guarded_cache_entry(model,state):
            for cache in (None,reference()):
                stream=model.generate_with_prompt_cache_streaming(**probe.generation('abc'),prompt_cache=cache)
                stream.close()
            self.assertEqual(len(model.calls),2)
            self.assertIsNone(model.calls[0]['prompt_cache'])
            self.assertEqual(model.calls[1]['prompt_cache']['mode'],'reference')
            self.assertEqual(state['guard']['reserved_slots'],471)
            with self.assertRaises(ValueError):
                model.generate_with_prompt_cache_streaming(**probe.generation('x'*557),prompt_cache=reference())
            with self.assertRaises(ValueError):model._generate_with_prompt_cache('bypass')
            self.assertEqual(len(model.calls),2)
        self.assertNotIn('_generate_with_prompt_cache',vars(model))

    def test_tuple_audio_closes_upstream_on_eof_cancel_and_bad_tuple(self):
        for cancel,bad in [(0,False),(1,False),(0,True)]:
            with self.subTest(cancel=cancel,bad=bad):
                model=Model(items=[('bad',)] if bad else None)
                pcm,row,_=probe.collect(model,probe.generation('abc'),None,cancel_after=cancel)
                self.assertEqual(model.stream.closed,1)
                self.assertTrue(row['generator_close_called'])
                self.assertEqual(row['outcome'],'failed' if bad else 'cancelled' if cancel else 'ok')
                self.assertEqual(row['normal_flush_completed'],not (cancel or bad))
                if not bad:self.assertTrue(pcm)

    def test_close_propagates_when_consumer_fails_and_hook_restores_on_exception(self):
        model=Model();original=lambda **kwargs:ClosingStream(model.items)
        model._generate_with_prompt_cache=original
        with self.assertRaisesRegex(RuntimeError,'consumer'):
            with probe.guarded_cache_entry(model,{}):
                iterator=probe.cached_audio(model,probe.generation('abc'),reference())
                next(iterator)
                upstream=iterator.gi_frame.f_locals['stream']
                try:raise RuntimeError('consumer')
                finally:iterator.close()
        self.assertEqual(upstream.closed,1)
        self.assertIs(model._generate_with_prompt_cache,original)

    def test_capacity_loader_removes_temporary_zero_hook_and_leaves_original_dispatch(self):
        model=Model();events=[]
        @contextmanager
        def hooks(slots,extra,state):
            self.assertEqual((slots,extra),(1024,2));events.append('enter')
            try:yield
            finally:events.append('exit')
        def load(*_):model._generate=lambda:'temporary';return model,{'ok':True}
        with patch.object(probe.capacity,'capacity_hooks',side_effect=hooks),patch.object(probe.native,'load_model',side_effect=load):
            actual,_=probe.load_with_capacity(None,None,None,None,{})
        self.assertIs(actual,model);self.assertEqual(events,['enter','exit'])
        self.assertNotIn('_generate',vars(model));self.assertEqual(model._generate(),'original')

    def test_schedule_has_three_excluded_and_three_measured_calls_per_case_mode(self):
        rows=probe.trials()
        for case in probe.CASES:
            for mode in ('A','B'):
                for condition in ('warmup','warm'):
                    reps=[r[3] for r in rows if r[:3]==(condition,case,mode)]
                    self.assertEqual(reps,[0,1,2])
        measured=[r for r in rows if r[0]=='warm']
        self.assertEqual([r[2] for r in measured[:2]],['A','B'])
        self.assertEqual([r[2] for r in measured[6:8]],['B','A'])


if __name__=='__main__':unittest.main()
