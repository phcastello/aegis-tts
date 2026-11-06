"""Tiny CPU Torch probes only: no pretrained models, audio, weights or CUDA."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import v2_fastpitch_ipa_backend as backend

torch.set_num_threads(2)


class AcousticProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 100)
        self.register_buffer('float_buffer', torch.ones(1, dtype=torch.float32))
        self.register_buffer('integer_buffer', torch.ones(1, dtype=torch.int64))
        self.output_dtype = None

    def infer(self, seq, pace=1.):
        value = self.linear(torch.ones(1, 3, 4, dtype=torch.float32))
        self.output_dtype = value.dtype
        return value.transpose(1, 2), torch.tensor([3], dtype=torch.int64)


class VocoderProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 256)
        self.register_buffer('float_buffer', torch.ones(1, dtype=torch.float32))
        self.input_dtype = self.output_dtype = None

    def forward(self, mel):
        self.input_dtype = mel.dtype
        value = self.linear(mel.transpose(1, 2))
        self.output_dtype = value.dtype
        return value.reshape(1, 1, -1)


def tiny_backend():
    value = backend.Backend.__new__(backend.Backend)
    value.torch, value.np, value.device = torch, np, 'cpu'
    value.precision = 'bf16'
    value.max_tokens, value.max_frames = 64, 16
    value.acoustic, value.vocoder = AcousticProbe().eval(), VocoderProbe().eval()
    value.model_module = SimpleNamespace(regulate_len=lambda *args: None)
    return value


class PrecisionContractTests(unittest.TestCase):
    def test_outer_bf16_autocast_cannot_change_vocoder_compute_dtype(self):
        value = tiny_backend()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            audio = value.generate_ids([1, 14, 2])
        self.assertEqual(value.acoustic.output_dtype, torch.bfloat16)
        self.assertEqual(value.vocoder.input_dtype, torch.float32)
        self.assertEqual(value.vocoder.output_dtype, torch.float32)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 3 * 256)
        self.assertTrue(np.isfinite(audio).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_non_fp32_floating_parameters_are_rejected_before_inference(self):
        for name in ('acoustic', 'vocoder'):
            with self.subTest(model=name):
                value = tiny_backend()
                getattr(value, name).linear.to(dtype=torch.bfloat16)
                with self.assertRaisesRegex(RuntimeError, 'FP32'):
                    value.generate_ids([1, 14, 2])
                self.assertIsNone(value.acoustic.output_dtype)

    def test_non_fp32_floating_buffers_are_rejected_before_inference(self):
        for name in ('acoustic', 'vocoder'):
            with self.subTest(model=name):
                value = tiny_backend()
                model = getattr(value, name)
                model.float_buffer = model.float_buffer.to(torch.float64)
                with self.assertRaisesRegex(RuntimeError, 'FP32'):
                    value.generate_ids([1, 14, 2])
                self.assertIsNone(value.acoustic.output_dtype)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
