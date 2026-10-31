"""Exact mastering regression from the approved scalar detector."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from voice_pipeline import StatefulM02


def test_m02_preserves_approved_double_precision_output_and_state() -> None:
    rng = np.random.default_rng(40733133)
    samples = rng.integers(-32768, 32768, 18000, dtype=np.int32).astype(np.float64) / 32768
    samples[4000:5000] = 0
    master = StatefulM02(24000)
    output = np.concatenate([master.process(piece) for piece in np.array_split(
        samples, [1, 37, 4000, 4999, 5000, 7999, 12000])])
    # Recorded before changing NumPy scalar iteration to native doubles.
    assert hashlib.sha256(output.astype("<f8").tobytes()).hexdigest() == (
        "1797e088d801dff249bfd549c4941f6179c4036838f88237a428dd320d427c5f")
    assert float(master._envelope).hex() == "0x1.98f1ef5facf64p-2"
