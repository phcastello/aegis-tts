"""CPU proof: arbitrary chunking is batch-equivalent and always closes native streams."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

from training.v2_voxcpm_streaming_probe import HalfRateFIR,collect_stream,summarize,trials
from training import v2_voxcpm_streaming_probe as probe


class StreamingResamplerTests(unittest.TestCase):
    def test_irregular_chunks_exactly_match_batch_and_flush_length(self):
        rng=np.random.default_rng(40733133)
        for length in [0,1,2,3,19,20,21,40,41,42,511,7680,10003]:
            with self.subTest(length=length):
                values=rng.normal(0,.1,length).astype(np.float32)
                resampler=HalfRateFIR();pieces=[];start=0
                for size in [1,0,3,17,2,41,7,89,4095,3841,10000]:
                    chunk=values[start:start+size];start+=len(chunk)
                    pieces.append(resampler.push(chunk))
                    self.assertLessEqual(len(resampler.history),41)
                self.assertEqual(start,length)
                pieces.append(resampler.flush());actual=np.concatenate(pieces)
                expected=resample_poly(values,1,2,window=('kaiser',10.)) if length else values
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(len(actual),(length+1)//2)
                with self.assertRaises(RuntimeError):resampler.push(np.zeros(1))
                with self.assertRaises(RuntimeError):resampler.flush()

    def test_impulse_phase_and_no_premature_output(self):
        values=np.zeros(103,dtype=np.float32);values[0]=.8;values[57]=-.5
        r=HalfRateFIR()
        self.assertEqual(len(r.push(values[:20])),0)
        pieces=[r.push(values[20:21]),r.push(values[21:]),r.flush()]
        np.testing.assert_array_equal(np.concatenate(pieces),resample_poly(values,1,2,window=('kaiser',10.)))

    def test_bad_channels_and_nonfinite_values_are_rejected(self):
        for values in [np.ones((2,10)),np.array([np.nan],dtype=np.float32)]:
            with self.assertRaises(ValueError):HalfRateFIR().push(values)


class FakeClock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,value):self.now+=value


class StreamingClientTests(unittest.TestCase):
    def test_local_arrivals_include_cpu_work_and_normal_flush(self):
        clock=FakeClock();closed=[];sink_chunks=[]
        values=(.2*np.sin(2*np.pi*500*np.arange(4047)/48000)).astype(np.float32)
        def factory():
            try:
                start=0
                for size in (7,41,0,3999):
                    clock.advance(.1);yield values[start:start+size];start+=size
            finally:closed.append(True)
        def convert(chunk):clock.advance(.001);return chunk
        def sink(pcm):sink_chunks.append(pcm);clock.advance(.002)
        pcm,row=collect_stream(factory,clock=clock,sink=sink,convert=convert)
        reference=np.rint(resample_poly(values,1,2,window=('kaiser',10.))*32767).astype('<i2').tobytes()
        self.assertEqual(pcm,reference);self.assertEqual(b''.join(sink_chunks),pcm)
        self.assertEqual(row['outcome'],'ok');self.assertEqual(closed,[True])
        self.assertTrue(row['normal_flush_completed']);self.assertTrue(row['generator_close_called'])
        self.assertEqual(row['pcm_chunks'][-1]['source'],'normal_eof_flush')
        first_native_nonempty=row['native_chunks'][1]['ready_ms']
        self.assertGreater(row['first_pcm_ms'],first_native_nonempty)
        self.assertGreater(row['eof_ms'],row['native_eof_ms'])
        self.assertGreater(row['underruns_over_10ms'],0)
        self.assertIsNotNone(row['ttfa_estimated_playback_ms'])

    def test_cancel_closes_generator_discards_tail_and_prevents_next_model_step(self):
        consumed=[];closed=[]
        def factory():
            try:
                for index in range(3):consumed.append(index);yield np.full(3000,.1,dtype=np.float32)
            finally:closed.append(True)
        pcm,row=collect_stream(factory,cancel_after_native_chunks=1)
        self.assertEqual(consumed,[0]);self.assertEqual(closed,[True]);self.assertEqual(row['outcome'],'cancelled')
        self.assertFalse(row['normal_flush_completed']);self.assertTrue(row['generator_close_called'])
        self.assertTrue(row['partial_pcm_delivered']);self.assertEqual(len(pcm)//2,1490)
        self.assertFalse(any(c['source']=='normal_eof_flush' for c in row['pcm_chunks']))

    def test_external_cancel_before_first_step_still_closes_generator(self):
        class Native:
            def __init__(self):self.steps=0;self.closed=False
            def __iter__(self):return self
            def __next__(self):self.steps+=1;return np.zeros(1000,dtype=np.float32)
            def close(self):self.closed=True
        stream=Native();pcm,row=collect_stream(lambda:stream,cancelled=lambda:True)
        self.assertEqual(stream.steps,0);self.assertTrue(stream.closed);self.assertEqual(pcm,b'')
        self.assertEqual(row['outcome'],'cancelled')

    def test_error_and_sink_error_both_close_stream(self):
        for bad_sink in (False,True):
            closed=[]
            def factory():
                try:yield np.full(3000,np.nan if not bad_sink else .1,dtype=np.float32)
                finally:closed.append(True)
            def sink(pcm):raise OSError('test sink closed')
            _,row=collect_stream(factory,sink=sink if bad_sink else None)
            self.assertEqual(row['outcome'],'failed');self.assertEqual(closed,[True])
            self.assertTrue(row['generator_close_called']);self.assertFalse(row['normal_flush_completed'])

    def test_three_excluded_warmups_for_every_case(self):
        order=trials(['v_reply','v_pressure'],3,True)
        self.assertEqual(len(order),12)
        for case in ('v_reply','v_pressure'):
            self.assertEqual(sum(c==case and kind!='warm' for kind,c,rep in order),3)
        self.assertEqual(order[0][0],'compile-first')
        self.assertEqual(order[3][0],'case-first')

    def test_short_rtf_is_not_an_architecture_rejection_without_long_gate(self):
        records=[]
        for case,rtf in [('short',1.5),('long',.9)]:
            for _ in range(3):records.append(dict(case=case,condition='warm',outcome='ok',
                first_pcm_ms=200,first_speech_pcm_available_ms=250,ttfa_estimated_playback_ms=220,
                eof_ms=1000,rtf=rtf,audio_seconds=1.,underrun_total_ms=0,underruns_over_10ms=0,
                potential_length_cap_hit=False))
        summary=summarize(records,{'short':{'size':'short'},'long':{'size':'long'}})
        self.assertIsNone(summary['short']['mandatory_long_rtf_gate_passed'])
        self.assertFalse(summary['short']['rejected_by_short_rtf_alone'])
        self.assertTrue(summary['long']['mandatory_long_rtf_gate_passed'])

    def test_default_plan_keeps_512mib_compile_cache_without_model_execution(self):
        output=io.StringIO()
        with patch('sys.argv',['streaming-probe']),contextlib.redirect_stdout(output),\
             patch.object(probe,'execute') as execution:
            probe.main()
        execution.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['compiler_cache_cap_mib'],512)


if __name__=='__main__':unittest.main()
