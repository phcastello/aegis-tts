import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from training import v2_voxcpm_lora_benchmark as bench


class LoRABenchmarkContracts(unittest.TestCase):
    def test_checkpoint_file_and_sidecar_are_bound_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'last.pt';path.write_bytes(b'tiny metadata fixture, not weights')
            receipt=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,step=4)
            path.with_suffix('.sha256.json').write_text(json.dumps(receipt))
            actual=bench.checkpoint_receipt(path)
            self.assertEqual((actual['sha256'],actual['step']),(receipt['sha256'],4))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'integrity'):bench.checkpoint_receipt(path)

    def test_actual_unmerged_branches_are_frozen_bf16_and_entirely_cuda(self):
        class LoRALinear:pass
        parameter=SimpleNamespace(dtype='BF16',device=SimpleNamespace(type='cuda'),requires_grad=False,numel=lambda:16)
        model=SimpleNamespace(named_parameters=lambda:[('layer.lora_A',parameter)],modules=lambda:[LoRALinear()])
        torch=SimpleNamespace(bfloat16='BF16')
        self.assertEqual(bench.lora_residency(model,torch)['elements'],16)
        self.assertFalse(bench.lora_residency(model,torch)['merged'])
        parameter.dtype='FP32'
        with self.assertRaisesRegex(ValueError,'BF16'):bench.lora_residency(model,torch)
        parameter.dtype='BF16';parameter.device.type='cpu'
        with self.assertRaisesRegex(ValueError,'CUDA'):bench.lora_residency(model,torch)
        parameter.device.type='cuda';model.modules=lambda:[]
        with self.assertRaisesRegex(ValueError,'unmerged'):bench.lora_residency(model,torch)


if __name__=='__main__':unittest.main()
