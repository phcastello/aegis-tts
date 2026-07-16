"""Small, dependency-light metrics helpers for Aegis TTS.

The service intentionally reports the PyTorch allocator numbers that belong to
this process.  ``nvidia-smi`` is optional and only supplements that view with
the global GPU state; absence of its executable never breaks synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any


MIB = 1024 * 1024


def _round_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / MIB, 2)


@dataclass(frozen=True)
class GpuMemory:
    """A process-local CUDA allocator snapshot, expressed in MiB."""

    allocated_mib: float | None = None
    reserved_mib: float | None = None
    peak_allocated_mib: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "allocated_mib": self.allocated_mib,
            "reserved_mib": self.reserved_mib,
            "peak_allocated_mib": self.peak_allocated_mib,
        }


class MetricsCollector:
    """Collect CUDA data without making CUDA a module-import requirement."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    @staticmethod
    def _torch() -> Any | None:
        try:
            import torch

            return torch
        except Exception:  # pragma: no cover - only relevant in broken installs
            return None

    def cuda_available(self) -> bool:
        torch = self._torch()
        try:
            return bool(torch and torch.cuda.is_available())
        except Exception:
            return False

    def reset_peak(self) -> None:
        """Reset allocator peak immediately before one serialized generation."""

        torch = self._torch()
        if not torch or not self.cuda_available():
            return
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            # Device indexing differs across some torch releases.  Metrics are
            # best effort and must never take the inference path down.
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                return

    def memory(self) -> GpuMemory:
        torch = self._torch()
        if not torch or not self.cuda_available():
            return GpuMemory()
        try:
            return GpuMemory(
                allocated_mib=_round_mib(torch.cuda.memory_allocated(self.device)),
                reserved_mib=_round_mib(torch.cuda.memory_reserved(self.device)),
                peak_allocated_mib=_round_mib(torch.cuda.max_memory_allocated(self.device)),
            )
        except Exception:
            return GpuMemory()

    def empty_cache(self) -> None:
        """Release unused allocator blocks after a handled CUDA OOM."""

        torch = self._torch()
        if not torch or not self.cuda_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            return

    def gpu_info(self) -> dict[str, Any]:
        """Return safe GPU summary fields for ``/health``.

        Values from ``nvidia-smi`` represent total machine usage while allocator
        values are this process only.  Keeping them named differently avoids
        conflating the two in operational dashboards.
        """

        data: dict[str, Any] = {
            "cuda_available": self.cuda_available(),
            "process": self.memory().as_dict(),
        }
        if not data["cuda_available"]:
            return data

        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return data
            values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
            if len(values) != 5:
                return data
            name, total, used, temperature, utilization = values
            data["global"] = {
                "name": name,
                "memory_total_mib": _as_number(total),
                "memory_used_mib": _as_number(used),
                "temperature_c": _as_number(temperature),
                "utilization_percent": _as_number(utilization),
            }
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
        return data


def _as_number(value: str) -> int | float | str:
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def real_time_factor(generation_seconds: float, audio_seconds: float) -> float | None:
    """Return RTF, avoiding a misleading division for empty model output."""

    if audio_seconds <= 0:
        return None
    return round(generation_seconds / audio_seconds, 4)
