"""CPU contract checks: real call arguments/metadata and per-case warmup."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.v2_voxcpm_optimized_probe import build_trials


class VoxCPMAblationTests(unittest.TestCase):
    def test_each_shape_is_warmed_before_matrix_observations(self):
        cases=['v_reply','v_pressure','v_long']
        schedule=build_trials(cases,3,True)
        first_warm=next(i for i,item in enumerate(schedule) if item[0]=='warm')
        prefix=schedule[:first_warm]
        self.assertEqual(first_warm,9)
        for case in cases:
            self.assertEqual(len([r for r in prefix if r[1]==case]),3)
            self.assertEqual([r[2] for r in schedule[first_warm:] if r[1]==case],[0,1,2])
        self.assertEqual(schedule[0][0],'compile-first')
        self.assertEqual(schedule[3][0],'case-first')
        self.assertEqual(schedule[6][0],'case-first')

    def test_duplicate_cases_cannot_overwrite_paired_evidence(self):
        with self.assertRaises(ValueError):build_trials(['v_reply','v_reply'],3,False)

    def test_generator_receives_selected_nfe_and_preserved_defaults(self):
        import numpy as np
        from training.v2_voxcpm_probe import synthesize
        class FakeModel:
            def parameters(self):return []
            def eval(self):return self
            def generate(self,**kw):
                self.kw=kw
                return (.1*np.sin(np.arange(9600,dtype=np.float32)*.2)).astype(np.float32)
        cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                             max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
        fake_torch=SimpleNamespace(cuda=cuda)
        with tempfile.TemporaryDirectory() as temp:
            for nfe in (None,10,6,4):
                args=SimpleNamespace(text='Sim, a válvula azul está fechada.',run=Path(temp))
                if nfe is not None:args.nfe=nfe
                model=FakeModel();row=synthesize(args,model,fake_torch,np,str(nfe))
                expected=10 if nfe is None else nfe
                self.assertEqual(model.kw['inference_timesteps'],expected)
                self.assertEqual(row['nfe'],expected)
                self.assertEqual(model.kw['cfg_value'],2.)
                self.assertEqual(model.kw['seed'],40733133)
                self.assertFalse(model.kw['retry_badcase'])
                self.assertEqual(row['outcome'],'ok')


if __name__=='__main__':unittest.main()
