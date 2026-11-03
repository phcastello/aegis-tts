"""CPU integration: paired optimizer continuation, crash publication and MAS."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_piper_experiment as piper

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch CPU unavailable')
class Recovery(unittest.TestCase):
    def setUp(self):
        import torch
        import numpy as np
        self.t=torch;self.np=np;torch.set_num_threads(1)

    def backend(self):
        t=self.t
        return SimpleNamespace(torch=t,np=self.np,device='cpu',g=t.nn.Linear(1,1,bias=False),d=t.nn.Linear(1,1,bias=False),
                               hp={},provenance=dict(fixture='tiny actual-autograd only'),saved_state=None)

    def args(self,root,updates,resume=None):
        root.mkdir(exist_ok=True);manifest=root.parent/'input.json';manifest.write_text('tiny fixture')
        return SimpleNamespace(action='pilot',run=root,seed=40733133,lr_gen=.001,lr_disc=.001,batch_size=1,accumulation=1,val_limit=1,patience=12,
                               min_delta=.005,cache_mib=0,data_manifest=manifest,resume=resume,max_updates=updates,checkpoint_seconds=1800)

    def run_tiny(self,a,b):
        data=dict(rows=dict(train=[dict(id=str(i)) for i in range(3)],validation=[dict(id='v')]),fingerprint='fixture',probes=[])
        def batch(rows,backend,cache):
            return self.t.tensor([[1+int(rows[0]['id'])+random.random()+self.np.random.random()+float(self.t.rand(()))]])
        def loss(backend,x):
            generated=backend.g(x);g=((backend.d(generated)-1)**2).mean();d=(backend.d(generated.detach())**2).mean()
            return g,d,dict(mel=float(g.detach()))
        with patch.object(piper,'budget',return_value={}),patch.object(piper,'make_batch',side_effect=batch),patch.object(piper,'compute_loss',side_effect=loss),\
             patch.object(piper,'validation',return_value=.5),patch.object(piper,'probes',return_value=[]),patch.object(piper,'emit'):
            piper.train(a,b,data)
        return piper.load_own(a.run/'last.pt',b)

    def test_three_pairs_save_fresh_resume_four_matches_uninterrupted_four(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.t.manual_seed(11);first=self.backend();initial=deepcopy((first.g.state_dict(),first.d.state_dict()))
            saved3=self.run_tiny(self.args(root/'resumed',3),first)
            self.assertEqual(saved3['step'],3)
            for optimizer in saved3['optimizers']:
                self.assertEqual({float(v['step']) for v in optimizer['state'].values()},{3.})
            resumed=self.backend();saved4=self.run_tiny(self.args(root/'resumed',4,root/'resumed/last.pt'),resumed)
            full=self.backend();full.g.load_state_dict(initial[0]);full.d.load_state_dict(initial[1]);continuous=self.run_tiny(self.args(root/'full',4),full)
            for kind in ('g','d'):
                for name,value in saved4[kind].items():self.assertTrue(self.t.equal(value,continuous[kind][name]))
            for o1,o2 in zip(saved4['optimizers'],continuous['optimizers']):
                for ident,value in o1['state'].items():
                    for field in ('step','exp_avg','exp_avg_sq'):self.assertTrue(self.t.equal(value[field],o2['state'][ident][field]))
            self.assertEqual((saved4['step'],saved4['epoch'],saved4['cursor'],saved4['permutation']),(4,continuous['epoch'],continuous['cursor'],continuous['permutation']))
            self.assertTrue(self.t.equal(saved4['rng']['torch'],continuous['rng']['torch']))
            self.assertEqual(saved4['rng']['python'],continuous['rng']['python'])
            self.assertTrue(self.np.array_equal(saved4['rng']['numpy'][1],continuous['rng']['numpy'][1]))
            self.assertFalse(self.t.cuda.is_initialized())

    def test_weight_publication_crash_reconciles_pending_and_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);b=self.backend();a=self.args(root/'run',3);old=self.run_tiny(a,b);oldhash=piper.sha(a.run/'best.pt')
            future=self.run_tiny(self.args(root/'future',4),self.backend())
            writer=piper.atomic_json
            def interrupted(path,value):
                if Path(path).name=='last.pt.json':raise RuntimeError('simulated power loss afterweightreplace')
                return writer(path,value)
            with patch.object(piper,'budget',return_value={}),patch.object(piper,'atomic_json',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'power loss'):piper.save_checkpoint(a,b,future)
            restored=piper.load_own(a.run/'last.pt',b)
            self.assertEqual(restored['step'],4);self.assertFalse((a.run/'last.pt.pending.json').exists())
            self.assertEqual(piper.sha(a.run/'best.pt'),oldhash)
            self.assertEqual(json.loads((a.run/'last.pt.json').read_text())['sha256'],piper.sha(a.run/'last.pt'))

    def test_missing_adam_moment_and_wrong_pair_step_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            b=self.backend();state=self.run_tiny(self.args(Path(temporary)/'run',3),b)
            bad=deepcopy(state);key=next(iter(bad['optimizers'][1]['state']));del bad['optimizers'][1]['state'][key]['exp_avg_sq']
            with self.assertRaisesRegex(ValueError,'moments'):piper.audit_state(bad,self.t)
            bad=deepcopy(state);bad['step']=4
            with self.assertRaisesRegex(ValueError,'Adam step'):piper.audit_state(bad,self.t)
            bad=deepcopy(state);bad['permutation']=[0,0,2]
            with self.assertRaisesRegex(ValueError,'permutation'):piper.audit_state(bad,self.t)
            bad=deepcopy(state);bad['best']=float('nan')
            with self.assertRaisesRegex(ValueError,'scalar'):piper.audit_state(bad,self.t)

    def test_preserve_rng_restores_python_numpy_torch(self):
        b=self.backend();before=piper.rng(self.t,self.np,'cpu')
        with piper.preserve_rng(b,99):random.random();self.np.random.random();self.t.rand(5)
        after=piper.rng(self.t,self.np,'cpu')
        self.assertEqual(before['python'],after['python']);self.assertTrue(self.np.array_equal(before['numpy'][1],after['numpy'][1]));self.assertTrue(self.t.equal(before['torch'],after['torch']))

    def test_real_mas_bridge_orientation_and_impossible_alignment_guard(self):
        path=Path('/var/tmp/aegis-v2-work/vits/deps/monotonic_alignment_search/core.cpython-312-x86_64-linux-gnu.so')
        if not path.exists():self.skipTest('pinned CP312 extension absent')
        import sys
        piper.install_mas(self.t,self.np);mas=sys.modules['piper.train.vits.monotonic_align']
        score=self.t.tensor([[[5.,0.,0.],[5.,0.,0.],[0.,5.,0.],[0.,5.,0.],[0.,0.,5.],[0.,0.,5.]]])
        actual=mas.maximum_path(score,self.t.ones_like(score))
        self.assertEqual(actual.argmax(2).tolist(),[[0,0,1,1,2,2]])
        with self.assertRaisesRegex(ValueError,'dimensions'):mas.maximum_path(self.t.ones(1,2,3),self.t.ones(1,2,3))

@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('soundfile'),'audio CPU dependencies unavailable')
class AudioQC(unittest.TestCase):
    def test_diagnostic_audio_keeps_onset_offsets_without_fictitious_latency(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            row=piper.audio_artifact(Path(temporary)/'speech.wav',np.full(22050,.05,dtype='float32'),SimpleNamespace(np=np))
            self.assertEqual(row['outcome'],'ok');self.assertEqual(row['onset_audio_offset_ms'],0.)
            self.assertNotIn('ttfa_estimated_playback_ms',row);self.assertNotIn('first_pcm_ms',row)
            self.assertEqual(row['clipped_samples'],0)

if __name__=='__main__':unittest.main()
