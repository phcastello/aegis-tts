"""IPA binding regressions; no public acoustic payload, WAV or GPU is loaded."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'training/v2_fastpitch_ipa_experiment.py'
OLD = REPO / 'training/fastpitch_experiment.py'
AUDIT = REPO / 'training/v2_fastpitch_ipa_checkpoint_audit.py'
OLD_AUDIT = REPO / 'training/v2_fastpitch_checkpoint_audit.py'


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


class Tensor:
    def __init__(self, values): self.values=list(values)
    def tolist(self): return list(self.values)


def binding_namespace():
    tree=functions(SOURCE)
    vocab=list(range(161))
    glyph=lambda text:Tensor([2]+[10+ord(c)%50 for c in text]+[2])
    torch=types.SimpleNamespace(tensor=lambda values,dtype=None:Tensor(values),long='int64',
                                equal=lambda a,b:a.values==b.values)
    ns=dict(VOCAB=vocab,torch=torch,Path=Path,
            grapheme_source=types.SimpleNamespace(tokens=glyph,normalize=lambda text:text.lower()),
            ipa_frontend=types.SimpleNamespace(MAP_SHA256='a'*64))
    selected=[tree[n] for n in ('validate_phone_ids','tokens','inject_phone_tokens')]
    exec(compile(ast.Module(body=selected,type_ignores=[]),'ipa_binding_only','exec'),ns)
    return ns


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.n=binding_namespace();glyph=self.n['grapheme_source'].tokens('oi').tolist()
        self.feature=dict(id='sample',feature='/tmp/ipa-synthetic-feature.pt',feature_sha256='b'*64,
                          frames=100,text_length=len(glyph),group_id='train-origin')
        self.row=dict(id='sample',source_text='oi',acoustic_feature_path=self.feature['feature'],
                      acoustic_feature_sha256='b'*64,frames=100,group_id='train-origin',
                      original_grapheme_text_length=len(glyph),original_grapheme_ids=glyph,
                      glyph_normalized_text='oi',audio_sha256='c'*64,audio_filepath='/tmp/ipa-synthetic.wav',
                      phoneme_ids=[1,14,141,2],phone_length=4,map_sha256='a'*64)
        self.raw=dict(id='sample',text=Tensor(glyph),normalized_text='oi',audio_sha256='c'*64,
                      audio_filepath='/tmp/ipa-synthetic.wav',mel=object(),pitch=object())

    def bind(self): return self.n['inject_phone_tokens'](self.raw,self.feature,self.row)

    def test_ram_override_keeps_original_tensor_and_acoustics(self):
        before=self.raw['text'].tolist();value=self.bind()
        self.assertEqual(value['text'].tolist(),[1,14,141,2])
        self.assertEqual(self.raw['text'].tolist(),before)
        self.assertIs(value['mel'],self.raw['mel']);self.assertIs(value['pitch'],self.raw['pitch'])

    def test_tampered_cached_glyph_is_rejected_before_phone_override(self):
        self.raw['text']=Tensor([99])
        with self.assertRaises(ValueError):self.bind()

    def test_tampered_declared_glyph_is_rejected(self):
        self.row['original_grapheme_ids']=[99]
        with self.assertRaises(ValueError):self.bind()

    def test_feature_audio_map_and_origin_mismatches_are_rejected(self):
        for key in ('acoustic_feature_sha256','audio_sha256','map_sha256','group_id'):
            with self.subTest(key=key):
                old=self.row[key];self.row[key]='wrong'
                with self.assertRaises(ValueError):self.bind()
                self.row[key]=old

    def test_phone_length_must_fit_the_actual_mel(self):
        self.row['phoneme_ids']=[1]+[14]*100+[2];self.row['phone_length']=102
        with self.assertRaises(ValueError):self.bind()

    def test_pad_interior_boundaries_bool_and_oov_are_rejected(self):
        for ids in ([1,0,2],[1,1,2],[1,2,2],[1,True,2],[1,161,2],[1,-1,2],'texto'):
            with self.subTest(ids=ids),self.assertRaises(ValueError):self.n['validate_phone_ids'](ids)


class SourceAndConfigTests(unittest.TestCase):
    def test_checkpoint_math_and_publisher_are_unchanged(self):
        old,new=functions(OLD),functions(SOURCE)
        names=[name for name in old if name.startswith('_checkpoint_')]
        names+=['imports','sample_batch','rng_state','restore_rng','mel_spectrogram']
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),
                         '7e7c94d8106b2e7729df5e238195f554dbcb5a494a3c5ee179f7ef8ee1a90709')

    def test_generic_audit_checks_are_retained(self):
        old,new=functions(OLD_AUDIT),functions(AUDIT)
        for name in ('adam_layout','compare_recipe'):
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
        def keys(tree):
            return {n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript)
                    and isinstance(n.value,ast.Name) and n.value.id=='checks'
                    and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
        self.assertTrue(keys(old['audit_once'])<=keys(new['audit_once']))

    def test_cache78_model161_rule_preserves_all_other_feature_metadata(self):
        fn=functions(AUDIT)['acoustic_config_matches'];ns={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'config_rule','exec'),ns)
        raw=dict(vocab=['glyph']*78,mel=dict(sr=24000),pitch_mean=137.)
        config=dict(raw_source_features=copy.deepcopy(raw),features=dict(raw,vocab=list(range(161))))
        self.assertTrue(ns['acoustic_config_matches'](raw,config,list(range(161))))
        for target in ('features','raw_source_features'):
            altered=copy.deepcopy(config);altered[target]['mel']['sr']=22050
            self.assertFalse(ns['acoustic_config_matches'](raw,altered,list(range(161))))

    def test_max_steps_is_not_a_scientific_config_field(self):
        train=functions(SOURCE)['train']
        config_writes=[n for n in ast.walk(train) if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets)]
        self.assertEqual(len(config_writes),1)
        self.assertNotIn('max_steps',ast.unparse(config_writes[0]))

    def test_fresh_probe_requires_saved_manifest_and_loader_bindings(self):
        fn=functions(SOURCE)['require_frozen_probe_manifest'];ns=dict(Path=Path)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_probe_binding','exec'),ns)
        path='/tmp/ipa-synthetic-frozen-manifest.json';digest='a'*64;loader='b'*64
        bound=dict(phone_manifest_path=path,phone_manifest_sha256=digest,phone_loader_code_sha256=loader)
        check=ns['require_frozen_probe_manifest'];check(bound,path,digest,loader)
        for args in [('/tmp/other-ipa-manifest.json',digest,loader),
                     (path,'c'*64,loader),(path,digest,'c'*64)]:
            with self.subTest(args=args),self.assertRaises(ValueError):check(bound,*args)
        for missing in bound:
            changed=dict(bound);changed.pop(missing)
            with self.subTest(missing=missing),self.assertRaises(ValueError):check(changed,path,digest,loader)
        body=functions(SOURCE)['synthesize']
        calls=[n.func.id for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('require_frozen_probe_manifest',calls)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'optional tiny CPU continuation needs Torch and NumPy')
class TinyCPUContinuation(unittest.TestCase):
    def test_new_schema_survives_save_restore_and_real_next_update(self):
        script=r'''
import copy,random,tempfile
from pathlib import Path
import numpy as np
import torch
from training import v2_fastpitch_ipa_experiment as f
torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42)
def setup():
 m=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Dropout(.2),torch.nn.Linear(3,2))
 o=torch.optim.AdamW(m.parameters(),lr=.0002,betas=(.9,.98),eps=1e-8,weight_decay=1e-6)
 s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:min(1.,(step+1)/1000))
 return m,o,s
def update(m,o,s):
 o.zero_grad();loss=m(torch.randn(2,4)).square().sum();loss.backward();o.step();s.step()
m,o,s=setup();update(m,o,s)
state=dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),scaler={},
 rng=f.rng_state(torch.device('cpu')),step=1,epoch=0,cursor=16,best=1.,bad=0,
 config=dict(precision='fp32',lr=.0002,warmup=1000,selection=dict(max_lr_reductions=1,plateau_lr_factor=.3),
             features=dict(vocab=f.VOCAB),raw_source_features=dict(vocab=f.grapheme_source.VOCAB),frontend=dict(schema=f.IPA_SCHEMA)),
 provenance=dict(code_sha256=f.digest(f.__file__),source_manifest_sha256='synthetic',upstream='synthetic',
                 torch=torch.__version__,cuda='CPU-only',argv=['synthetic-test'],utc='synthetic'),
 lr_state=dict(multiplier=1.,reductions=0))
with tempfile.TemporaryDirectory() as t:
 f._checkpoint_publish(Path(t),state,m,True)
 update(m,o,s);expected=copy.deepcopy(m.state_dict())
 m2,o2,s2=setup();loaded,_=f._checkpoint_resume(Path(t),m2)
 assert loaded['config']['frontend']['schema']==f.IPA_SCHEMA and len(loaded['config']['features']['vocab'])==161
 m2.load_state_dict(loaded['model']);o2.load_state_dict(loaded['optimizer']);s2.load_state_dict(loaded['scheduler'])
 f.restore_rng(loaded['rng'],torch.device('cpu'));update(m2,o2,s2)
 assert all(torch.equal(expected[k],v) for k,v in m2.state_dict().items())
 assert s2.last_epoch==2 and not torch.cuda.is_initialized()
print('tiny CPU continuation PASS; no public model/data/GPU')
'''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        ret=subprocess.run([sys.executable,'-B','-c',script],cwd=REPO,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(ret.returncode,0,ret.stdout+ret.stderr)
        self.assertIn('continuation PASS',ret.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
