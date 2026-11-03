"""Small CPU autograd recovery and validation-state regression tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training import v2_voxcpm_train as trainer


class DataContracts(unittest.TestCase):
    def test_original_nfc_is_used_and_test_partition_is_never_opened(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split,ident in (('train','a'),('validation','b')):
                row=dict(id=ident,text='processed text',original_text='A\u0301udio '+ident,duration=3.,
                         source_audio_filepath='/private/'+ident,source_audio_sha256=ident,group_id=ident)
                (root/(split+'.jsonl')).write_text(json.dumps(row)+'\n')
            (root/'test.jsonl').write_bytes(b'INVALID HELDOUT MUST NOT BE PARSED')
            args=SimpleNamespace(prepared=root,pilot_longest=False,train_limit=0,validation_limit=0)
            data=trainer.load_data(args)
            self.assertEqual(data['rows']['train'][0]['_text'],'Áudio a')
            self.assertEqual(set(data['rows']),{'train','validation'})


@unittest.skipUnless(importlib.util.find_spec('torch') and not (trainer.ROOT/'pause-asr').exists(),
                     'Needs optional Torch and no measured GPU timing window')
class Recovery(unittest.TestCase):
    def setUp(self):
        import torch
        import numpy as np
        self.t,self.np=torch,np;torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.lora_A=torch.nn.Parameter(torch.tensor(.4));self.lora_B=torch.nn.Parameter(torch.tensor(.2));self.cache=torch.ones(())
            def forward(self,value):
                result=value*self.lora_A+self.cache*self.lora_B
                return {'loss/diff':((result-1)**2).mean(),'loss/stop':(.01*(result+1)**2).mean()}
        self.Tiny=Tiny

    def backend(self):
        return SimpleNamespace(t=self.t,np=self.np,model=self.Tiny(),device='cpu',max_sequence=1024)

    def args(self,path,steps,resume=None):
        path.mkdir(exist_ok=True)
        return SimpleNamespace(run=path,rank=8,lr=.001,warmup=2,seed=7311026,accumulation=2,
            max_steps=steps,validate_every=1,checkpoint_seconds=1800,patience=8,min_delta=.001,
            max_sequence=1024,audio_cache_mib=0,saved_tensors_cpu=False,resume=resume,
            resume_allow_resource_change=False,resume_allow_code_change=False,nfe=4,
            pilot_longest=False,train_limit=3,validation_limit=1,probe_limit=1)

    def run_tiny(self,args,b):
        data=dict(rows={'train':[dict(id=str(i)) for i in range(3)],'validation':[dict(id='v')]},fingerprint='tiny-fixture',manifest={'fixture':'no audio/model weights'})
        def batch(row,backend,_cache):
            value=1+int(row['id'])+random.random()+float(self.np.random.rand())+float(self.t.rand(()))
            return [self.t.tensor(value)],1
        def validate(args,backend,*_):
            with trainer.validation_context(backend,11):
                random.random();self.np.random.rand();self.t.rand(2)
            return .5
        def probes(args,backend,step):
            with trainer.validation_context(backend,19):
                with self.t.inference_mode():
                    backend.model.cache=self.t.ones(())
                    self.t.rand(3)
            return {'fixture':True}
        with patch.object(trainer,'budget',return_value={}),patch.object(trainer,'make_batch',side_effect=batch),\
             patch.object(trainer,'validation',side_effect=validate),patch.object(trainer,'native_probes',side_effect=probes),\
             patch.object(trainer,'emit'),patch.object(trainer,'cleanup_probes'):
            trainer.train(args,b,data)
        return trainer.load_checkpoint(args.run/'last.pt',self.t)

    def assert_rng_equal(self,a,b):
        self.assertEqual(a['python'],b['python'])
        self.assertEqual(a['numpy'][0],b['numpy'][0]);self.assertTrue(self.np.array_equal(a['numpy'][1],b['numpy'][1]))
        self.assertEqual(a['numpy'][2:],b['numpy'][2:]);self.assertTrue(self.t.equal(a['torch'],b['torch']))
        self.assertEqual(len(a['cuda']),len(b['cuda']))

    def test_three_update_checkpoint_fresh_resume_four_matches_uninterrupted_four(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);before=self.backend()
            saved3=self.run_tiny(self.args(root/'resumed',3),before)
            self.assertEqual(saved3['step'],3)
            fresh=self.backend();saved4=self.run_tiny(self.args(root/'resumed',4,'last'),fresh)
            continuous=self.run_tiny(self.args(root/'continuous',4),self.backend())
            for name,value in saved4['lora'].items():self.assertTrue(self.t.equal(value,continuous['lora'][name]))
            for ident,values in saved4['optimizer']['state'].items():
                for field in ('step','exp_avg','exp_avg_sq'):
                    self.assertTrue(self.t.equal(values[field],continuous['optimizer']['state'][ident][field]))
            self.assertEqual((saved4['step'],saved4['epoch'],saved4['cursor'],saved4['permutation']),
                             (continuous['step'],continuous['epoch'],continuous['cursor'],continuous['permutation']))
            self.assert_rng_equal(saved4['rng'],continuous['rng'])
            self.assertFalse(self.t.cuda.is_initialized())

    def test_validation_repairs_inference_cache_and_preserves_all_training_rng(self):
        b=self.backend();before=trainer.rng(self.t,self.np,'cpu')
        with trainer.validation_context(b,901):
            random.random();self.np.random.rand();self.t.rand(4)
            with self.t.inference_mode():b.model.cache=self.t.ones(())
            self.assertTrue(self.t.is_inference(b.model.cache))
        self.assertFalse(self.t.is_inference(b.model.cache))
        value=trainer.loss([self.t.tensor(2.)],b);value.backward()
        self.assertIsNotNone(b.model.lora_B.grad)
        self.assertTrue(self.t.isfinite(b.model.lora_B.grad))
        self.assert_rng_equal(before,trainer.rng(self.t,self.np,'cpu'))

    def test_save_is_rng_neutral_and_pending_publication_recovers_after_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b=self.backend();args=self.args(root/'run',3)
            old=self.run_tiny(args,b);best_hash=trainer.pilot.sha(args.run/'best.pt')
            future=self.run_tiny(self.args(root/'future',4),self.backend())
            before=trainer.rng(self.t,self.np,'cpu')
            writer=trainer.atomic_json
            def crash(path,value):
                if Path(path).name=='last.sha256.json':raise RuntimeError('simulated crash after weight replacement')
                return writer(path,value)
            with patch.object(trainer,'budget',return_value={}),patch.object(trainer,'atomic_json',side_effect=crash):
                with self.assertRaisesRegex(RuntimeError,'simulated crash'):trainer.save_checkpoint(args,b,future)
            restored=trainer.load_checkpoint(args.run/'last.pt',self.t)
            self.assertEqual(restored['step'],4)
            self.assertFalse((args.run/'last.pending.json').exists())
            self.assertEqual(trainer.pilot.sha(args.run/'best.pt'),best_hash)
            self.assert_rng_equal(before,trainer.rng(self.t,self.np,'cpu'))

    def test_partial_adam_bad_permutation_and_missing_rng_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            state=self.run_tiny(self.args(Path(temp)/'run',3),self.backend())
            broken=deepcopy(state);ident=next(iter(broken['optimizer']['state']))
            del broken['optimizer']['state'][ident]['exp_avg_sq']
            with self.assertRaisesRegex(ValueError,'Adam moments'):trainer.audit_state(broken,self.t)
            broken=deepcopy(state);broken['permutation']=[0,0,2]
            with self.assertRaisesRegex(ValueError,'permutation'):trainer.audit_state(broken,self.t)
            broken=deepcopy(state);del broken['rng']['cuda']
            with self.assertRaisesRegex(ValueError,'RNG'):trainer.audit_state(broken,self.t)


if __name__=='__main__':unittest.main()
