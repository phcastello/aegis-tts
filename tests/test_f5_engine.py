"""CPU contract checks for reproducible validation and resumable fine-tuning."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

from training.f5_engine import Engine, parser, require_gpu_authorization


@unittest.skipIf(torch is None, "requires the private F5 training environment")
class F5EngineContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "prepared/mels").mkdir(parents=True)
        np.save(self.root / "prepared/mels/one.npy", np.ones((100, 12), dtype=np.float32))

    def tearDown(self):
        self.tmp.cleanup()

    def engine(self):
        class TinyFlow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, inp, text, lens):
                loss = ((self.weight * inp - 0.4) ** 2).mean()
                loss = loss + torch.rand(()) * 0.01 + random.random() * 0.02
                return loss, inp, inp * self.weight

        obj = Engine.__new__(Engine)
        obj.args = parser().parse_args([])
        obj.root = self.root
        obj.device = "cpu"
        obj.torch = torch
        obj.model = TinyFlow()
        obj.optimizer = torch.optim.AdamW(obj.model.parameters(), lr=1e-5, foreach=False)
        obj.scaler = torch.amp.GradScaler("cuda", enabled=False)
        obj.checkpoints = self.root / "checkpoints"
        obj.checkpoints.mkdir(exist_ok=True)
        obj.memory = lambda: {}
        obj.rows = {"validation": [dict(id="one", _tokens=list("teste"), duration=1.0)]}
        obj.dataset_fingerprint = "fixture"
        obj.update = 0
        obj.best = obj.reference_best = obj.initial_validation = 1.0
        obj.best_epoch = 0
        obj.patience = 0
        obj.lr_reduced = False
        obj.phase = "pilot"
        obj.elapsed_training = 0.0
        obj.history = []
        obj.peak_allocated_mib = obj.peak_reserved_mib = 0.0
        return obj

    def test_validation_reuses_noise_and_restores_python_and_torch_rng(self):
        engine = self.engine()
        random.seed(1337)
        torch.manual_seed(123)
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        first = engine.validate()
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        second = engine.validate()
        self.assertEqual(first["per_item_losses"], second["per_item_losses"])
        self.assertTrue(engine.model.training)

    def test_resume_replays_next_update_with_optimizer_and_rng(self):
        engine = self.engine()
        row = engine.rows["validation"][0]
        random.seed(37)
        torch.manual_seed(74)
        with contextlib.redirect_stdout(io.StringIO()):
            engine.train_group([row, row], epoch=1)
            engine.save(1, 1)
            expected = engine.train_group([row, row], epoch=1)
            weight = engine.model.weight.detach().clone()
            resumed = self.engine()
            self.assertEqual(resumed.resume(), (1, 1))
            actual = resumed.train_group([row, row], epoch=1)
        self.assertEqual(expected["loss"], actual["loss"])
        self.assertEqual(expected["gradient_norm"], actual["gradient_norm"])
        self.assertEqual(expected["lr"], actual["lr"])
        self.assertTrue(torch.equal(weight, resumed.model.weight))
        self.assertEqual(resumed.update, 2)

    def test_pilot_extends_to_six_without_improvement(self):
        engine = self.engine()
        self.assertIsNone(engine.assess_validation(3, 1.0)[1])
        self.assertIn("six_epochs", engine.assess_validation(6, 1.0)[1])

    def test_plateau_reduces_learning_rate_then_stops_after_more_validation(self):
        engine = self.engine()
        engine.phase = "full"
        engine.best = engine.reference_best = 0.8
        for epoch in range(12, 18):
            with contextlib.redirect_stdout(io.StringIO()):
                _, reason, _ = engine.assess_validation(epoch, 0.8)
            self.assertIsNone(reason)
        self.assertTrue(engine.lr_reduced)
        for epoch in range(18, 23):
            self.assertIsNone(engine.assess_validation(epoch, 0.8)[1])
        self.assertIn("converged", engine.assess_validation(23, 0.8)[1])

    def test_progress_can_continue_at_arbitrarily_late_epoch(self):
        engine = self.engine()
        engine.best = 0.7
        with contextlib.redirect_stdout(io.StringIO()):
            meaningful, reason, _ = engine.assess_validation(10000, 0.7)
        self.assertTrue(meaningful)
        self.assertIsNone(reason)
        self.assertEqual(engine.phase, "full")

    def test_authorization_required_before_gpu_query(self):
        args = parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "CUDA requires"):
            require_gpu_authorization(args, self.root)


if __name__ == "__main__":
    unittest.main()
