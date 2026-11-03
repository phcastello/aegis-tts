"""Offline timing-contract checks; no service/model/GPU access."""
import array
import unittest
from benchmarks.v2.benchmark import pcm_metrics

class PcmTimingTest(unittest.TestCase):
    def pcm(self,values):
        a=array.array('h',values)
        import sys
        if sys.byteorder!='little':a.byteswap()
        return a.tobytes()
    def test_no_speech_for_silence_or_one_frame_click(self):
        for values in ([0]*1440,[0]*480+[8000]*480+[0]*480):
            pcm=self.pcm(values)
            m=pcm_metrics(pcm,[dict(ms=100,bytes=len(pcm))])
            self.assertIsNone(m['first_speech_pcm_available_ms'])
    def test_late_chunk_respects_initial_silence_and_arrival(self):
        pcm=self.pcm([0]*480+[1000]*960)
        m=pcm_metrics(pcm,[dict(ms=100,bytes=960),dict(ms=200,bytes=1920)])
        self.assertEqual(m['first_pcm_ms'],100)
        self.assertEqual(m['first_speech_pcm_available_ms'],200)
        self.assertEqual(m['ttfa_estimated_playback_ms'],200)
        self.assertEqual(m['onset_audio_offset_ms'],20)
        self.assertEqual(m['onset_margin_audio_offset_ms'],0)
        self.assertEqual(m['underruns_over_10ms'],1)
        self.assertEqual(m['underrun_total_ms'],80)
    def test_sample_split_across_bytes_is_not_audio_early(self):
        pcm=self.pcm([1000]*960)
        m=pcm_metrics(pcm,[dict(ms=10,bytes=1),dict(ms=40,bytes=len(pcm)-1)])
        self.assertEqual(m['first_pcm_ms'],40)
        self.assertEqual(m['first_speech_pcm_available_ms'],40)
        self.assertEqual(m['ttfa_estimated_playback_ms'],40)
    def test_rejects_truncated_or_invalid_schedule(self):
        with self.assertRaises(ValueError):pcm_metrics(b'\0',[dict(ms=0,bytes=1)])
        with self.assertRaises(ValueError):pcm_metrics(b'\0\0',[dict(ms=2,bytes=1),dict(ms=1,bytes=1)])
        with self.assertRaises(ValueError):pcm_metrics(b'\0\0',[])
    def test_whole_file_audio_waits_for_file_then_silence(self):
        pcm=self.pcm([0]*2400+[1000]*960)
        m=pcm_metrics(pcm,[dict(ms=700,bytes=len(pcm))])
        self.assertEqual(m['first_speech_pcm_available_ms'],700)
        self.assertEqual(m['ttfa_estimated_playback_ms'],800)
        self.assertEqual(m['ttfa_margin_playback_ms'],780)

if __name__=='__main__':unittest.main()
