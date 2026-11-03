"""CPU integration checks: file-ready timing, failed audio and paired ASR texts."""
import importlib.util
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from training import v2_fastpitch_benchmark as bench


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('soundfile'),'audio CPU dependencies unavailable')
class FileBenchmarkIntegration(unittest.TestCase):
    def test_timing_includes_frontend_model_and_file_close_and_keeps_distinct_texts(self):
        import numpy as np
        clock=[0.];calls=[];case=dict(id='number',size='short',text='Teste 2.')
        class Backend:
            fp=SimpleNamespace(normalize=lambda s:s.lower());max_frames=1000
            def synchronize(self):pass
            def reset_peak(self):pass
            def peaks(self):return {}
            def generate(self,text):
                calls.append(('backend',text));clock[0]+=.2
                return np.full(24000,.1,dtype='float32')
        def normalize(text):calls.append(('frontend',text));clock[0]+=.1;return {'tts_text':'Teste dois.'}
        original=bench.write_pcm
        def write_and_close(path,pcm):
            original(path,pcm)
            with wave.open(str(path),'rb') as wav:self.assertEqual(wav.getnframes(),24000)
            calls.append(('closed',str(path)));clock[0]+=.05
        with tempfile.TemporaryDirectory() as temporary,patch.object(bench,'write_pcm',side_effect=write_and_close):
            row=bench.measure_case(Backend(),case,normalize,bench.metrics_module(),Path(temporary),'candidate',0,clock=lambda:clock[0])
            self.assertEqual(row['outcome'],'ok');self.assertAlmostEqual(row['eof_ms'],350.)
            self.assertEqual([c[0] for c in calls],['frontend','backend','closed'])
            self.assertEqual(row['effective_text'],'teste dois.');self.assertTrue(Path(row['wav']).exists())
            job=bench.asr_job(case,row,{'number':'Teste dois.'})
            self.assertEqual((job['expected_verbatim'],job['expected_spoken'],job['expected_effective']),('Teste 2.','Teste dois.','teste dois.'))

    def test_overshoot_is_preserved_in_float_and_never_written_as_clipped_pcm(self):
        import numpy as np
        import soundfile as sf
        class Backend:
            fp=SimpleNamespace(normalize=lambda s:s);max_frames=1000
            def synchronize(self):pass
            def reset_peak(self):pass
            def peaks(self):return {}
            def generate(self,text):return np.full(24000,1.01,dtype='float32')
        with tempfile.TemporaryDirectory() as temporary,patch.object(bench,'write_pcm') as writer:
            row=bench.measure_case(Backend(),dict(id='overflow',size='short',text='Teste.'),lambda x:{'tts_text':x},bench.metrics_module(),Path(temporary),'candidate',0)
            self.assertEqual(row['outcome'],'failed');writer.assert_not_called();self.assertNotIn('wav',row)
            self.assertEqual(row['samples_at_generator_bound'],24000)
            raw,_=sf.read(row['raw_float'],dtype='float32');self.assertGreater(float(raw.max()),1.)

    def test_generator_saturation_count_survives_legal_pcm_conversion(self):
        import numpy as np
        class Backend:
            fp=SimpleNamespace(normalize=lambda s:s);max_frames=1000
            def synchronize(self):pass
            def reset_peak(self):pass
            def peaks(self):return {}
            def generate(self,text):return np.tile(np.array([1.,-1.,.9],dtype='float32'),8000)
        with tempfile.TemporaryDirectory() as temporary:
            row=bench.measure_case(Backend(),dict(id='saturated',size='short',text='Teste.'),bench.frontend('native'),bench.metrics_module(),Path(temporary),'candidate',0,frontend_mode='native')
            self.assertEqual(row['outcome'],'ok');self.assertEqual(row['outside_unit'],0)
            self.assertEqual(row['samples_at_generator_bound'],16000)
            self.assertAlmostEqual(row['fraction_at_generator_bound'],2/3)
            self.assertEqual(row['frontend_mode'],'native');self.assertEqual(row['effective_text'],'Teste.')

    def test_silent_pcm_is_a_no_speech_failure(self):
        import numpy as np
        class Backend:
            fp=SimpleNamespace(normalize=lambda s:s);max_frames=1000
            def synchronize(self):pass
            def reset_peak(self):pass
            def peaks(self):return {}
            def generate(self,text):return np.zeros(24000,dtype='float32')
        with tempfile.TemporaryDirectory() as temporary:
            row=bench.measure_case(Backend(),dict(id='silence',size='short',text='Teste.'),lambda x:{'tts_text':x},bench.metrics_module(),Path(temporary),'candidate',0)
            self.assertEqual(row['outcome'],'no_speech');self.assertIsNone(row['ttfa_estimated_playback_ms'])

    def test_same_mel_count_with_different_hop_is_incompatible(self):
        config=dict(features=dict(vocab=['a'],mel={'signature':'fixed'},pitch_mean=100.,pitch_std=30.),
                    model=dict(n_symbols=1,n_mel_channels=100),alignment_precision='FP32 cumulative duration boundaries and pitch sums; acoustic outputs retain selected dtype')
        vocoder=dict(sampling_rate=24000,n_fft=1024,hop_size=512,win_size=1024,num_mels=100,fmin=0,fmax=None,upsample_rates=[8,8,8])
        with self.assertRaisesRegex(ValueError,'incompatible'):bench.validate_pair(config,vocoder,['a'],{'signature':'fixed'})


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch CPU dependency unavailable')
class PredictedDurationGuard(unittest.TestCase):
    def test_guard_rejects_long_allocation_and_zero_duration_before_regulator(self):
        import torch
        calls=[];module=SimpleNamespace(regulate_len=lambda *a,**k:calls.append(True))
        original=module.regulate_len
        with bench.bounded_regulator(module,torch,100):
            for duration in (torch.tensor([[0.,0.]]),torch.tensor([[101.,1.]])):
                with self.assertRaises(ValueError):module.regulate_len(duration,torch.ones(1,2,4))
        self.assertFalse(calls);self.assertIs(module.regulate_len,original)

    def test_half_ties_match_actual_pinned_nonjit_regulator(self):
        import torch
        source=Path('/var/tmp/aegis-v2-work/fastpitch/source/fastpitch/model.py')
        if not source.exists():self.skipTest('pinned upstream source unavailable')
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='regulate_len')
        namespace=dict(torch=torch,F=torch.nn.functional,Optional=__import__('typing').Optional)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        module=SimpleNamespace(regulate_len=namespace['regulate_len'])
        durations=torch.tensor([[.5,1.5,2.5]]);encoded=torch.arange(3).float().view(1,3,1)
        with bench.bounded_regulator(module,torch,6):
            actual,length=module.regulate_len(durations,encoded)
        self.assertEqual(length.tolist(),[6]);self.assertEqual(actual.flatten().tolist(),[0.,1.,1.,2.,2.,2.])
        with bench.bounded_regulator(module,torch,5):
            with self.assertRaisesRegex(ValueError,'exceeds'):module.regulate_len(durations,encoded)


if __name__=='__main__':unittest.main()
