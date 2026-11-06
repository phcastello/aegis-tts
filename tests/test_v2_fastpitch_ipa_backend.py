"""Stdlib protocol/guard tests; no Torch, WAV, weights or GPU operations."""
from __future__ import annotations
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training import v2_fastpitch_ipa_backend as backend
from training import v2_fastpitch_ipa_frontend as front


def fake_command(mutation='', preamble='', after_ready='', after_response=''):
    md = front.frontend_metadata()
    md.update(torch_imported=False, native_initialized=True)
    ready = dict(status='ready', frontend=md, vocab_sha256=front.VOCAB_SHA256)
    script = 'import json,sys,time\n' + preamble + '\n'
    script += 'print(' + repr(json.dumps(ready)) + ',flush=True)\n' + after_ready + '\n'
    script += 'for line in sys.stdin:\n r=json.loads(line)\n'
    script += ' out=dict(id=r["id"],ok=True,normalized_text=r["text"],sentences=[["a"]],ids=[1,14,2],map_sha256=' + repr(front.MAP_SHA256) + ',config_sha256=' + repr(front.CONFIG_SHA256) + ',vocab_sha256=' + repr(front.VOCAB_SHA256) + ',frontend_version=' + repr(front.VERSION) + ')\n'
    if mutation:
        script += ' ' + mutation + '\n'
    script += ' print(json.dumps(out),flush=True)\n'
    if after_response:
        script += ' ' + after_response + '\n'
    return [sys.executable, '-S', '-B', '-u', '-c', script]


class BackendProtocolTests(unittest.TestCase):
    def client(self, **kwargs):
        value = backend.FrontendClient(**kwargs)
        self.addCleanup(value.close)
        return value

    def test_default_plan_never_opens_payload_or_imports_torch(self):
        # Other collected tests may legitimately import Torch in the parent.
        # This metadata-only invocation must preserve, not erase, that state.
        torch_present_before = 'torch' in sys.modules
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix.lower() in ('.wav', '.pt', '.ckpt', '.onnx', '.safetensors'):
                raise AssertionError('unexpected payload: ' + str(path))
            return original(path, *args, **kwargs)
        output = io.StringIO()
        with patch.object(Path, 'open', guarded), contextlib.redirect_stdout(output):
            self.assertEqual(backend.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertFalse(value['checkpoint_or_audio_payload_opened'])
        self.assertEqual(value['torch_imported'], torch_present_before)
        self.assertEqual('torch' in sys.modules, torch_present_before)

    def test_two_requests_use_one_owned_child_and_strict_ids(self):
        client = self.client(command=fake_command())
        pid = client.process.pid
        a = client.phonemize('a'); b = client.phonemize('a')
        self.assertEqual(client.process.pid, pid)
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(a['ids'], [1,14,2])
        client.close()
        self.assertIsNotNone(client.process.poll())
        with self.assertRaisesRegex(backend.FrontendError, 'closed'):
            client.phonemize('a')

    def test_wrong_id_fails_and_reaps_only_owned_child(self):
        unrelated = subprocess.Popen([sys.executable, '-S', '-c', 'import time;time.sleep(30)'])
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.terminate())
        self.addCleanup(lambda: unrelated.wait(timeout=3))
        client = self.client(command=fake_command('out["id"]="other"'))
        with self.assertRaisesRegex(backend.FrontendError, 'ID mismatch'):
            client.phonemize('a')
        self.assertIsNotNone(client.process.poll())
        self.assertIsNone(unrelated.poll())
        unrelated.terminate(); unrelated.wait(timeout=3)

    def test_response_map_mismatch_is_not_accepted_or_retried(self):
        client = self.client(command=fake_command('out["map_sha256"]="0"*64'))
        with self.assertRaisesRegex(backend.FrontendError, 'pin mismatch'):
            client.phonemize('a')
        self.assertEqual(client.counter, 1)
        self.assertIsNotNone(client.process.poll())

    def test_response_unknown_phone_and_inconsistent_ids_fail(self):
        for mutation in ('out["sentences"]=[["☃"]]', 'out["ids"]=[1,161,2]', 'out["ids"]=[1,14,0,2]'):
            with self.subTest(mutation=mutation):
                client = self.client(command=fake_command(mutation))
                with self.assertRaises((ValueError, backend.FrontendError)):
                    client.phonemize('a')
                self.assertIsNotNone(client.process.poll())

    def test_explicit_native_error_has_no_hidden_retry(self):
        client = self.client(command=fake_command('out=dict(id=r["id"],ok=False,error="native OOV")'))
        with self.assertRaisesRegex(backend.FrontendError, 'native OOV'):
            client.phonemize('a')
        self.assertEqual(client.counter, 1)
        self.assertIsNotNone(client.process.poll())

    def test_request_timeout_closes_and_reaps_child(self):
        client = self.client(command=fake_command('time.sleep(5)'), timeout=.1)
        with self.assertRaisesRegex(backend.FrontendError, 'timeout'):
            client.phonemize('a')
        self.assertIsNotNone(client.process.poll())

    def test_eof_mid_response_does_not_accept_partial_json(self):
        client = self.client(command=fake_command('sys.stdout.write("{partial");sys.stdout.flush();sys.exit(0)'))
        with self.assertRaises(backend.FrontendError):
            client.phonemize('a')
        self.assertIsNotNone(client.process.poll())

    def test_stdout_and_stderr_hard_bounds_terminate_owned_child(self):
        for kwargs in (
            dict(command=fake_command('sys.stdout.write("x"*9000);sys.stdout.flush();time.sleep(5)'), max_line_bytes=8192),
            dict(command=fake_command('sys.stderr.write("x"*9000);sys.stderr.flush();time.sleep(5)'), max_stderr_bytes=8192),
        ):
            with self.subTest(kwargs=tuple(kwargs)):
                client = self.client(**kwargs)
                with self.assertRaises(backend.FrontendError):
                    client.phonemize('a')
                self.assertIsNotNone(client.process.poll())
                self.assertLessEqual(len(client.stderr_tail), client.max_stderr_bytes)

    def test_text_and_token_bounds_are_checked_before_neural_work(self):
        client = self.client(command=fake_command(), max_text_bytes=2)
        with self.assertRaisesRegex(backend.FrontendError, 'byte bound'):
            client.phonemize('aaaa')
        for ids in ([1,14,2,2], [1,True,2], [1,0,2], [1,161,2], [1,14], [1,14,3,14,2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                backend.validate_ids(ids, 3)

    def test_normalization_is_explicit_refinement_then_training_NFC_lower(self):
        native = backend.normalize_for_frontend('  Mãe—PÃO…  ', 'native')
        self.assertEqual(native['effective_text'], 'mãe-pão...')
        self.assertEqual(native['source_text'], '  Mãe—PÃO…  ')
        refined = backend.normalize_for_frontend('A CPU verificou a API às 14h05.', 'refinement')
        self.assertIn('cê pê u', refined['effective_text'])
        self.assertIn('a pê i', refined['effective_text'])
        self.assertTrue(refined['applied_rules'])
        for text in ('', '\x00a', '\ud800'):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                backend.normalize_for_frontend(text)

    def test_checkpoint_metadata_can_validate_without_training_files(self):
        mel = dict(sr=24000,n_fft=1024,hop=256,win=1024,n_mels=100,fmin=0.,fmax=None)
        raw = dict(vocab=backend.RAW_VOCAB,mel=mel,pitch_mean=200.,pitch_std=20.)
        binding = dict(schema=front.SCHEMA,phone_manifest_path='/missing-training/phones.json',
            phone_manifest_sha256='0'*64,symbols_sha256=front.VOCAB_SHA256,map_sha256=front.MAP_SHA256,
            frontend_code_sha256=backend.sha(front.__file__),source_features_path='/missing-training/features.json',
            source_features_sha256='0'*64,raw_grapheme_trainer_sha256=backend.RAW_TRAINER_SHA,
            phone_loader_code_sha256='0'*64)
        config = dict(raw_source_features=raw,features=dict(raw,vocab=list(front.SYMBOLS)),frontend=binding,
                      model=dict(n_symbols=161,n_mel_channels=100),alignment_precision=backend.ALIGNMENT_POLICY)
        h = dict(sampling_rate=24000,n_fft=1024,hop_size=256,win_size=1024,num_mels=100,fmin=0,fmax=None,upsample_rates=[8,8,2,2])
        fp = SimpleNamespace(VOCAB=list(front.SYMBOLS),MEL_CONFIG=mel)
        self.assertEqual(backend.validate_pair(config,h,fp)['vocab_sha256'],front.VOCAB_SHA256)
        bad=copy.deepcopy(config);bad['frontend']['map_sha256']='0'*64
        with self.assertRaises(ValueError):backend.validate_pair(bad,h,fp)
        bad=copy.deepcopy(config);bad['features']['pitch_mean']+=1
        with self.assertRaises(ValueError):backend.validate_pair(bad,h,fp)

    def test_actual_CPU_daemon_handles_new_PTBR_and_NFD_without_Torch(self):
        torch_present_before = 'torch' in sys.modules
        client = self.client(startup_timeout=20.)
        first = client.phonemize('mãe e pão, amanhã.')
        second = client.phonemize('a mãe põe o pão na mesa.')
        self.assertIn(141, first['ids'])
        self.assertEqual(first['ids'][0],1); self.assertEqual(first['ids'][-1],2)
        self.assertNotEqual(first['id'],second['id'])
        self.assertFalse(client.ready['frontend']['torch_imported'])
        self.assertEqual('torch' in sys.modules, torch_present_before)


class SmallTensor:
    def __init__(self, values):self.values=list(values)
    def float(self):return self
    def long(self):return SmallTensor(int(x) for x in self.values)
    def __truediv__(self,x):return SmallTensor(v/x for v in self.values)
    def __add__(self,x):return SmallTensor(v+x for v in self.values)
    def __lt__(self,x):return SmallTensor(v<x for v in self.values)
    def __le__(self,x):return SmallTensor(v<=x for v in self.values)
    def __gt__(self,x):return SmallTensor(v>x for v in self.values)
    def any(self):return any(self.values)
    def all(self):return all(self.values)
    def sum(self,dim):return SmallTensor([sum(self.values)])


class DurationBoundTests(unittest.TestCase):
    def test_zero_negative_nonfinite_overlong_duration_rejected_and_patch_restored(self):
        original=lambda *args: 'called'
        module=SimpleNamespace(regulate_len=original)
        torch=SimpleNamespace(isfinite=lambda x:SmallTensor(math.isfinite(v) for v in x.values))
        for values in ([0.,0.],[-1.,2.],[float('nan')],[11.], [6.,6.]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                with backend.bounded_regulator(module,torch,10):module.regulate_len(SmallTensor(values),None)
            self.assertIs(module.regulate_len,original)

    def test_valid_duration_and_pace_use_original_without_clamping(self):
        calls=[]
        def original(*args):calls.append(args);return 'original-result'
        module=SimpleNamespace(regulate_len=original)
        torch=SimpleNamespace(isfinite=lambda x:SmallTensor(math.isfinite(v) for v in x.values))
        durations=SmallTensor([2.,3.])
        with backend.bounded_regulator(module,torch,10):
            self.assertEqual(module.regulate_len(durations,None,pace=1.),'original-result')
        self.assertIs(calls[0][0],durations)
        self.assertIs(module.regulate_len,original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
