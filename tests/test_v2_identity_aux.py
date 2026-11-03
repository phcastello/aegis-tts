"""CPU fixtures for actual comparison failure modes, never speaker-score gates."""
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from training import v2_identity_aux as aux


class IdentityAuxTests(unittest.TestCase):
    def test_f5_effective_text_blocks_are_not_confused_with_a_string(self):
        self.assertEqual(aux.effective_text_sha(['primeira frase.','segunda frase.']),aux.text_sha('primeira frase. segunda frase.'))
        with self.assertRaises(ValueError): aux.effective_text_sha([{'text':'unsupported schema'}])

    def test_active_rms_ignores_appended_silence(self):
        # An otherwise equal voice with an added silent tail should not get
        # a larger analysis gain just because it pauses longer.
        t = np.arange(24000)/24000
        y = .2*np.sin(2*np.pi*120*t)
        self.assertAlmostEqual(aux.active_rms(y)[0],aux.active_rms(np.r_[y,np.zeros(24000)])[0],places=12)

    def test_shared_gain_target_never_clips_high_crest_factor(self):
        t = np.arange(24000)/24000
        a = .8*np.sin(2*np.pi*120*t); b = a*.01; b[100]=.99
        gains, policy = aux.matched_gains([a,b])
        self.assertLessEqual(max(np.abs(a*gains[0]).max(),np.abs(b*gains[1]).max()),.9500000001)
        self.assertAlmostEqual(aux.active_rms(a*gains[0])[0],aux.active_rms(b*gains[1])[0],places=10)
        self.assertFalse(policy['perceptual_loudness_lufs'])

    def test_silence_is_rejected_instead_of_fabricated_embedding_input(self):
        with self.assertRaisesRegex(ValueError,'No active audio'):
            aux.matched_gains([np.zeros(24000)])

    def test_dtw_refuses_different_text_even_if_wave_features_match(self):
        x = np.ones((10,3))
        with self.assertRaisesRegex(ValueError,'same requested text'):
            aux.dtw_same_requested_text(x,x,'different','sentences')

    def test_dtw_closed_end_does_not_hide_extra_wrong_tail(self):
        x = np.zeros((10,3)); extra = np.r_[x,np.ones((4,3))*8]
        match = aux.dtw_same_requested_text(x,x,'a','a')
        wrong_tail = aux.dtw_same_requested_text(x,extra,'a','a')
        self.assertEqual(match['mean_absolute_log_mel_per_path_step'],0)
        self.assertGreater(wrong_tail['mean_absolute_log_mel_per_path_step'],0)
        self.assertEqual(wrong_tail['frames'],[10,14])

    def test_dtw_budget_checked_before_allocating_quadratic_matrix(self):
        x = np.zeros((100,3))
        with self.assertRaisesRegex(ValueError,'cell budget'):
            aux.dtw_same_requested_text(x,x,'a','a',max_cells=1)

    def test_cpu_guard_obeys_product_timing_pause(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(aux,'ROOT',Path(d)):
            (Path(d)/'pause-asr').touch()
            with self.assertRaisesRegex(RuntimeError,'pause-asr'):
                aux.cpu_guard()


if __name__=='__main__': unittest.main()
