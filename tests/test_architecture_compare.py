"""Benchmark measurement/gating tests; no GPU, service, model or real network."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import httpx
import pytest

MODULE = Path(__file__).resolve().parents[1] / "benchmarks/architecture_compare/benchmark.py"
SPEC = importlib.util.spec_from_file_location("architecture_compare", MODULE)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def window():
    now = dt.datetime.now(dt.timezone.utc)
    return {"owner_approved": True, "authorization_id": "test-only", "candidate": "test",
            "starts_utc": (now - dt.timedelta(minutes=1)).isoformat(),
            "ends_utc": (now + dt.timedelta(minutes=1)).isoformat()}


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001/v1/audio/speech",
    "http://localhost:18001/tts", "https://127.0.0.1:18001/tts", "http://example.com:18001/tts",
    "http://127.0.0.1:18001/tts?token=secret", "http://user:pass@127.0.0.1:18001/tts"])
def test_endpoint_cannot_touch_production_or_external_runtime(url):
    with pytest.raises(ValueError):
        benchmark.validate_endpoint(url)
    assert benchmark.validate_endpoint("http://127.0.0.1:18001/tts")


def test_window_must_be_owner_approved_current_and_candidate_bound():
    valid = window()
    assert benchmark.validate_window(valid, "test") > 0
    with pytest.raises(ValueError, match="exact candidate"):
        benchmark.validate_window(valid, "another")
    with pytest.raises(ValueError, match="owner-approved"):
        benchmark.validate_window({**valid, "owner_approved": False}, "test")
    expired = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=2)
    with pytest.raises(ValueError, match="outside"):
        benchmark.validate_window(valid, "test", expired)
    with pytest.raises(ValueError, match="include UTC"):
        benchmark.validate_window({**valid, "starts_utc": "2026-01-01T12:00:00"}, "test")


def test_default_plan_does_not_probe_network_gpu_or_import_models(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "gpu_snapshot", lambda: pytest.fail("GPU used in plan"))
    monkeypatch.setattr(httpx.Client, "stream", lambda *a, **k: pytest.fail("POST in plan"))
    assert benchmark.main([]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["mode"] == "plan_only"
    assert observed["network_used"] is False and observed["gpu_used"] is False


def test_gpu_gate_refuses_shared_production_and_wrong_card():
    snapshot = {"processes": [{"pid": 12}, {"pid": 13}], "name": "RTX A2000 12GB", "total_mib": 11514}
    with pytest.raises(ValueError, match="process set"):
        benchmark.exclusive_gpu(snapshot, [12])
    benchmark.exclusive_gpu(snapshot, [12, 13])
    with pytest.raises(ValueError, match="A2000"):
        benchmark.exclusive_gpu({**snapshot, "total_mib": 24576}, [12, 13])


def test_useful_audio_requires_frames_and_honors_late_odd_chunks():
    pcm = b"\0\0" * 12000 + struct.pack("<h", 1000) * 480
    schedule = [{"ms": 300, "bytes": 3839}, {"ms": 301, "bytes": 1},
                {"ms": 1400, "bytes": len(pcm) - 3840}]
    row = benchmark.pcm_metrics(pcm, schedule, 24000)
    assert row["first_pcm_ms"] == 300
    assert row["first_useful_pcm_available_ms"] == 1400
    assert row["onset_audio_offset_ms"] == 500
    assert row["estimated_first_audible_playback_ms"] == pytest.approx(1820)
    assert row["immediate_playback_underruns_over_10ms"] == 1
    assert row["audio_seconds"] == .52
    silent = b"\0\0" * 480
    assert benchmark.pcm_metrics(silent, [{"ms": 10, "bytes": 960}], 24000)["first_useful_pcm_available_ms"] is None
    click = struct.pack("<h", -32768) + b"\0\0" * 479
    row = benchmark.pcm_metrics(click, [{"ms": 10, "bytes": 960}], 24000)
    assert row["first_useful_pcm_available_ms"] is None
    assert row["peak_pcm16"] == 32768
    assert row["clipped_samples"] == 1
    with pytest.raises(ValueError, match="truncated"):
        benchmark.pcm_metrics(b"\0", [{"ms": 10, "bytes": 1}], 24000)
    with pytest.raises(ValueError, match="does not consume"):
        benchmark.pcm_metrics(silent, [], 24000)


def test_receive_separates_transport_from_pcm_and_does_not_claim_true_streaming(monkeypatch):
    real_client = httpx.Client
    pcm = struct.pack("<h", 1000) * 480
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield pcm[:1]
            yield pcm[1:]
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Type": "audio/pcm"}, stream=Body()))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport))
    observed, row = benchmark.receive({"candidate": "test", "endpoint": "http://127.0.0.1:18001/tts", "sample_rate": 24000},
                                      window(), {"input": "test"}, SimpleNamespace(error=None))
    assert observed == pcm
    assert row["first_transport_byte_ms"] <= row["first_pcm_ms"] <= row["first_useful_pcm_available_ms"]
    assert row["rtf_http_complete"] >= 0
    assert "cannot prove incremental" in row["streaming_proof_limitation"]
    assert "true_streaming" not in row


def test_private_ab_preserves_native_pcm_and_category(tmp_path):
    directory = benchmark.private_directory(tmp_path / "review")
    a, b = tmp_path / "source-a.wav", tmp_path / "source-b.wav"
    a.write_bytes(benchmark.wav_bytes(b"\0\0" * 480, 24000))
    b.write_bytes(benchmark.wav_bytes(b"\0\0" * 640, 32000))
    rows = benchmark.build_ab({"pairs": [{"a": str(a), "b": str(b), "case": "test", "text": "Texto.",
                                          "category": "codec_reconstruction", "same_generation": True}]}, directory, 7)
    assert rows[0]["category"] == "codec_reconstruction"
    assert rows[0]["same_generation"] is True
    copies = {(directory / rows[0][label]).read_bytes() for label in ["A", "B"]}
    assert copies == {a.read_bytes(), b.read_bytes()}
    assert (directory / "index.html").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="shared text"):
        benchmark.build_ab({"pairs": [{"a": str(a), "b": str(b), "case": "test"}]}, directory, 7)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="outside any Git"):
        benchmark.private_directory(repo / "audio")


def test_offline_summary_keeps_failure_denominators_and_all_successful_outliers():
    rows = [{"candidate": "test", "case": "short", "outcome": "ok",
             "first_useful_pcm_available_ms": value, "eof_ms": value * 2, "rtf_http_complete": value / 1000}
            for value in [100, 200, 10000]]
    rows += [{"candidate": "test", "case": "short", "outcome": "failed", "error": "OOM", "eof_ms": 1},
             {"candidate": "test", "case": "long", "outcome": "no_useful_pcm", "eof_ms": 300}]
    result = benchmark.summarize(rows)
    aggregate = result["by_candidate_mixed_cases"][0]
    assert aggregate["observations"] == 5 and aggregate["successes"] == 3
    assert aggregate["failures"] == 2 and aggregate["failure_rate"] == .4
    assert aggregate["scope"] == "mixed_cases_descriptive_only"
    short = next(row for row in result["by_candidate_case"] if row["case"] == "short")
    assert short["failure_rate"] == .25
    assert short["metrics"]["eof_ms"]["p50"] == 400
    assert short["metrics"]["eof_ms"]["max"] == 20000
    assert next(row for row in result["by_candidate_case"] if row["case"] == "long")["metrics"]["eof_ms"]["p50"] is None


def test_summary_cli_is_offline_and_nearest_rank_p95_is_explicit(tmp_path, monkeypatch, capsys):
    rows = [{"candidate": "test", "case": "one", "outcome": "ok", "eof_ms": value} for value in range(1, 21)]
    source = tmp_path / "results.json"
    source.write_text(json.dumps(rows))
    monkeypatch.setattr(benchmark, "gpu_snapshot", lambda: pytest.fail("GPU used in summarize"))
    monkeypatch.setattr(httpx.Client, "stream", lambda *a, **k: pytest.fail("network used in summarize"))
    assert benchmark.main(["summarize", "--results", str(source)]) == 0
    observed = json.loads(capsys.readouterr().out)
    metric = observed["by_candidate_case"][0]["metrics"]["eof_ms"]
    assert metric["p50"] == 10.5 and metric["p95_nearest_rank"] == 19 and metric["max"] == 20
    assert "small samples" in observed["percentiles"]
