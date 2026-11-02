"""Portable archive privacy and overload preservation, without a neural runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave
import zipfile

import numpy as np
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parents[1] / "benchmarks/architecture_compare"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("package_listening", HERE / "package_listening.py")
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def test_float_overload_preserved_bit_for_bit(tmp_path):
    source, target = tmp_path / "original.wav", tmp_path / "portable.wav"
    sf.write(source, np.array([0.0, 1.2, -1.1, 0.3]), 24000, subtype="FLOAT")
    result = pack.listening_copy(source, target)
    assert result["pcm16"] is False
    assert result["samples_at_or_above_fullscale"] == 2
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("value", ["/var/tmp/private.wav", {"x": "failed at /home/alice/input.wav"},
                                   ["file:///private/audio"], "C:\\private\\audio.wav"])
def test_absolute_source_paths_rejected(value):
    with pytest.raises(ValueError, match="absolute paths"):
        pack.portable(value)


def test_archive_contains_all_attempts_and_only_explicit_inputs(tmp_path):
    root = tmp_path / "experiment"
    filenames = {
        "f5/roundtrip-treated-02": ["A-original.wav", "B-input-24000.wav", "C-vocos-24000.wav"],
        "magpie/roundtrip-treated-v1": ["A-original.wav", "B-input-22050-float.wav", "C-roundtrip-22050-float.wav"],
    }
    for directory, names in filenames.items():
        folder = root / directory
        folder.mkdir(parents=True)
        for name in names:
            with wave.open(str(folder / name), "wb") as audio:
                audio.setparams((1, 2, 24000, 10, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 10)
    (root / "f5/roundtrip-treated-02/results.json").write_text('{"safe": "codec receipt"}')
    (root / "magpie/roundtrip-treated-v1/receipt.json").write_text('{"safe": "codec receipt"}')
    synth = root / "f5/infer-novel-short-01"
    synth.mkdir()
    text = pack.read_json(HERE / "corpus.pt-BR.json")[0]["text"]
    for nfe, amplitude in [(8, 1.2), (16, 0.2)]:
        wav = synth / f"novel_short-nfe{nfe}-r00.wav"
        sf.write(wav, np.array([0.0, amplitude, -amplitude, 0.0]), 24000, subtype="FLOAT")
        wav.with_suffix(".json").write_text(json.dumps({
            "source_text": text, "nfe": nfe, "repetition": 0, "sha256": pack.digest(wav),
            "path": "/home/private/reference.wav",
        }))
    (root / ".env").write_text("PRIVATE_DO_NOT_COPY")
    (root / "weights.safetensors").write_text("PRIVATE_DO_NOT_COPY")
    result = pack.package(SimpleNamespace(experiment_root=root, output_dir=tmp_path / "portable",
                                         zip=tmp_path / "portable.zip", require_nfe=[8, 16], blind_seed=1))
    assert result["blind_pairs"] == 7  # Six codec controls plus both NFE attempts.
    assert {row["nfe"] for row in result["f5_observations"]} == {8, 16}
    with zipfile.ZipFile(result["zip"]) as archive:
        assert archive.testzip() is None
        assert not any("weights" in name or ".env" in name for name in archive.namelist())
        receipts = json.loads(archive.read(pack.HIDDEN + "/amostras-e-conversoes.json"))
        assert len(receipts) == 8
        assert sum(row["pcm16"] is False for row in receipts) == 1
        for name in archive.namelist():
            if name.endswith((".json", ".html", ".txt", ".csv")):
                data = archive.read(name).decode()
                assert "/home/" not in data and "PRIVATE_DO_NOT_COPY" not in data
        index = archive.read("escuta/index.html").decode()
        assert "downloadJSON" in index and "downloadCSV" in index
        assert "nfe8" not in index and "nfe16" not in index and "Vocos" not in index
