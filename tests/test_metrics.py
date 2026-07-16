from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.metrics import MIB, MetricsCollector, real_time_factor


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_device: str | None = None
        self.cache_cleared = False

    def is_available(self) -> bool:
        return True

    def reset_peak_memory_stats(self, device: str | None = None) -> None:
        self.reset_device = device

    def memory_allocated(self, device: str) -> int:
        return 128 * MIB

    def memory_reserved(self, device: str) -> int:
        return 256 * MIB

    def max_memory_allocated(self, device: str) -> int:
        return 384 * MIB

    def empty_cache(self) -> None:
        self.cache_cleared = True


def test_real_time_factor_is_safe_for_empty_audio() -> None:
    assert real_time_factor(2.5, 5.0) == 0.5
    assert real_time_factor(1.0, 0.0) is None


def test_cuda_allocator_metrics_are_reported_in_mib(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = _FakeCuda()
    monkeypatch.setattr(MetricsCollector, "_torch", staticmethod(lambda: SimpleNamespace(cuda=cuda)))
    collector = MetricsCollector("cuda:0")

    collector.reset_peak()
    memory = collector.memory()
    collector.empty_cache()

    assert cuda.reset_device == "cuda:0"
    assert memory.allocated_mib == 128
    assert memory.reserved_mib == 256
    assert memory.peak_allocated_mib == 384
    assert cuda.cache_cleared is True


def test_gpu_info_can_include_nvidia_smi_without_making_it_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _FakeCuda()
    monkeypatch.setattr(MetricsCollector, "_torch", staticmethod(lambda: SimpleNamespace(cuda=cuda)))
    monkeypatch.setattr(
        "app.metrics.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA RTX A2000, 6144, 512, 52, 17\n",
        ),
    )

    info = MetricsCollector().gpu_info()

    assert info["cuda_available"] is True
    assert info["process"]["allocated_mib"] == 128
    assert info["global"] == {
        "name": "NVIDIA RTX A2000",
        "memory_total_mib": 6144,
        "memory_used_mib": 512,
        "temperature_c": 52,
        "utilization_percent": 17,
    }

