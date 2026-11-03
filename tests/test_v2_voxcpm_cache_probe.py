import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training import v2_voxcpm_cache_probe as cache


class FakeTensor:
    dtype='torch.bfloat16'
    device='cuda'
    def __init__(self,layers,capacity):self.shape=(2,layers,1,2,capacity,128)
    def numel(self):return math.prod(self.shape)
    def element_size(self):return 2


class FakeModel:
    def __init__(self):
        self.config=SimpleNamespace(max_length=8192,lm_config=SimpleNamespace(max_position_embeddings=32768))
        self.base_lm=SimpleNamespace(forward_step=lambda:None)
        self.residual_lm=SimpleNamespace(forward_step=lambda:None)
        self.calls=[]
    def text_tokenizer(self,text):return list(text)
    def _generate(self,**kwargs):self.calls.append(kwargs);return ['generated']
    def generate(self,**kwargs):return next(iter(self._generate(streaming=False,**kwargs)))
    def generate_streaming(self,**kwargs):return iter(self._generate(streaming=True,**kwargs))


def fake_move(model,_torch):
    for name,layers in (('base_lm',28),('residual_lm',8)):
        getattr(model,name).kv_cache=SimpleNamespace(max_length=model.config.max_length,
                      kv_cache=FakeTensor(layers,model.config.max_length))
    return model


class CacheAblationTests(unittest.TestCase):
    def test_bound_includes_full_requested_generation_and_both_extra_slots(self):
        self.assertEqual(cache.capacity_guard(622,400,1024)['spare_slots'],0)
        with self.assertRaisesRegex(ValueError,'refuse truncation'):
            cache.capacity_guard(623,400,1024)
        for counts in ((1,400,1024,1),(1,-1,1024,2),(True,400,1024,2)):
            with self.assertRaises(ValueError):cache.capacity_guard(*counts)

    def test_both_caches_change_before_original_move_and_central_guard_covers_both_apis(self):
        model=FakeModel();state={}
        with patch.object(cache.pilot,'model_to_gpu',side_effect=fake_move):
            with cache.capacity_hooks(1024,2,state):
                moved=cache.pilot.model_to_gpu(model,None)
                self.assertIs(moved,model)
                self.assertEqual(state['native_capacity'],8192)
                self.assertEqual(state['actual']['caches']['base_lm']['capacity'],1024)
                self.assertEqual(state['actual']['caches']['residual_lm']['capacity'],1024)
                self.assertEqual(model.config.lm_config.max_position_embeddings,32768)
                self.assertEqual(model.generate(target_text='abc',max_len=400),'generated')
                self.assertEqual(list(model.generate_streaming(target_text='abc',max_len=400)),['generated'])
                self.assertEqual(len(model.calls),2)
                self.assertEqual(state['last_generation_guard']['reserved_slots'],405)
                for api in (model.generate,model.generate_streaming):
                    with self.assertRaises(ValueError):api(target_text='x'*623,max_len=400)
                    with self.assertRaises(ValueError):api(target_text='abc',max_len=400,prompt_text='reference')
                self.assertEqual(len(model.calls),2,'Rejected requests must never reach native generation')

    def test_process_hooks_restore_shared_helpers_even_after_exception(self):
        original=(cache.pilot.model_to_gpu,cache.pilot.synthesize,cache.optimized.residency,cache.native.collect_stream)
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with cache.capacity_hooks(1024,2,{}):raise RuntimeError('test failure')
        self.assertEqual(original,(cache.pilot.model_to_gpu,cache.pilot.synthesize,cache.optimized.residency,cache.native.collect_stream))

    def test_residency_refuses_one_cache_still_using_the_old_capacity(self):
        model=fake_move(FakeModel(),None)
        model.base_lm.kv_cache=SimpleNamespace(max_length=1024,kv_cache=FakeTensor(28,1024))
        with self.assertRaisesRegex(RuntimeError,'residual_lm'):
            cache.cache_record(model,1024)

    @unittest.skipIf((cache.optimized.ROOT/'pause-asr').exists() or importlib.util.find_spec('torch') is None,
                     'Needs optional torch and no measured GPU timing window')
    def test_tiny_cpu_masked_gqa_is_equal_for_valid_positions(self):
        result=cache.cpu_attention_proof()
        self.assertEqual(result['status'],'pass')
        self.assertFalse(result['cuda_initialized'])
        self.assertLessEqual(max(row['max_abs_difference'] for row in result['rows']),1e-12)


if __name__=='__main__':unittest.main()
