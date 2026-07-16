"""Private, local-only voice reference configuration and validation.

No route imports this module's transcript text.  It is read exclusively while
building the in-memory Qwen clone prompt at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
from pydantic import ValidationError

from .schemas import VoiceConfig, VoicesConfig

if TYPE_CHECKING:
    from .config import AppSettings


class ReferenceError(RuntimeError):
    """Base class for a configuration/reference error safe to show to clients."""


class ReferenceNotFoundError(ReferenceError):
    """Raised when a supplied voice id does not exist in configured profiles."""


class ReferenceValidationError(ReferenceError):
    """Raised when a configured source cannot safely create a clone prompt."""


@dataclass(frozen=True, slots=True)
class ReferenceMetadata:
    """Non-sensitive technical metadata; transcript content is intentionally absent."""

    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bits_per_sample: int | None = None
    codec: str | None = None
    size_bytes: int | None = None
    peak: float | None = None
    rms: float | None = None
    silence_ratio: float | None = None
    clipping_ratio: float | None = None
    audio_sha256: str | None = None
    transcript_sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        """Provide only fields allowed in public voice responses."""

        return {
            "reference_duration_seconds": self.duration_seconds,
            "reference_sample_rate": self.sample_rate,
        }


@dataclass(frozen=True, slots=True)
class VoiceReference:
    """A validated private input used for startup prompt creation."""

    config: VoiceConfig
    audio_path: Path
    transcript_path: Path
    transcript: str
    metadata: ReferenceMetadata


@dataclass(slots=True)
class ReferenceRepository:
    """Load trusted local configuration and validate reference assets once.

    The JSON config has lexical schema protection, then all resolved paths are
    constrained to ``settings.voices_dir``.  That second check covers symlinks
    and is what prevents a benign-looking config from escaping the mounted
    reference directory.
    """

    settings: "AppSettings"
    # Useful for unit tests and for an intentionally relocated private volume.
    # Production leaves this as None and uses settings.voices_dir (/voices).
    voices_root: Path | None = None
    _config: VoicesConfig | None = field(default=None, init=False, repr=False)
    _references: dict[str, VoiceReference] = field(default_factory=dict, init=False, repr=False)
    _reference_errors: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def load_config(self, *, reload: bool = False) -> VoicesConfig:
        if self._config is not None and not reload:
            return self._config

        config_path = Path(self.settings.voices_config)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReferenceValidationError("arquivo de configuração de vozes ausente") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceValidationError("arquivo de configuração de vozes inválido") from exc
        try:
            config = VoicesConfig.model_validate(raw)
        except ValidationError as exc:
            raise ReferenceValidationError("configuração de vozes não passou na validação") from exc

        enabled_ids = {voice.id for voice in config.voices if voice.enabled}
        if config.default_voice not in enabled_ids:
            raise ReferenceValidationError("a voz padrão precisa estar habilitada")
        self._config = config
        if reload:
            self._references.clear()
            self._reference_errors.clear()
        return config

    # Short aliases keep callers readable and make a deliberate reload explicit
    # without exposing transcript content.
    def load(self) -> VoicesConfig:
        return self.load_config()

    def refresh(self) -> VoicesConfig:
        return self.load_config(reload=True)

    @property
    def default_voice_id(self) -> str:
        return self.load_config().default_voice

    def voice_configs(self, *, enabled_only: bool = False) -> list[VoiceConfig]:
        voices = self.load_config().voices
        return [voice for voice in voices if voice.enabled] if enabled_only else list(voices)

    def list_references(self, *, validate: bool = True) -> dict[str, VoiceReference]:
        """Return valid enabled references, retaining a status for invalid ones."""

        references: dict[str, VoiceReference] = {}
        for voice in self.voice_configs(enabled_only=True):
            if not validate and voice.id in self._references:
                references[voice.id] = self._references[voice.id]
                continue
            try:
                references[voice.id] = self.get_reference(voice.id, validate=validate)
            except ReferenceValidationError as exc:
                self._reference_errors[voice.id] = _safe_error(exc)
        return references

    def get_reference(self, voice_id: str, *, validate: bool = True) -> VoiceReference:
        config = next((voice for voice in self.voice_configs() if voice.id == voice_id), None)
        if config is None:
            raise ReferenceNotFoundError("voice id desconhecido")
        if not config.enabled:
            raise ReferenceValidationError("voz desabilitada")
        if voice_id in self._references and not validate:
            return self._references[voice_id]
        if voice_id in self._references:
            return self._references[voice_id]

        audio_path = self._resolve_reference_path(config.audio_path)
        try:
            transcript_path = self._resolve_reference_path(config.transcript_path)
        except ReferenceValidationError as exc:
            raise ReferenceValidationError("transcrição de referência ausente") from exc
        try:
            transcript = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise ReferenceValidationError("transcrição de referência ausente") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ReferenceValidationError("transcrição de referência ilegível") from exc
        if not transcript.strip():
            raise ReferenceValidationError("transcrição de referência vazia")

        metadata = self._validate_audio(audio_path, transcript_path)
        reference = VoiceReference(
            config=config,
            audio_path=audio_path,
            transcript_path=transcript_path,
            transcript=transcript,
            metadata=metadata,
        )
        self._references[voice_id] = reference
        self._reference_errors.pop(voice_id, None)
        return reference

    def reference_error(self, voice_id: str) -> str | None:
        return self._reference_errors.get(voice_id)

    def status_by_voice(self) -> dict[str, str]:
        """State limited to an ID and outcome; never reveal local file paths."""

        states: dict[str, str] = {}
        for voice in self.voice_configs():
            if not voice.enabled:
                states[voice.id] = "disabled"
            elif voice.id in self._references:
                states[voice.id] = "validated"
            elif voice.id in self._reference_errors:
                states[voice.id] = "error"
            else:
                states[voice.id] = "pending"
        return states

    def _resolve_reference_path(self, configured_path: str) -> Path:
        candidate = Path(configured_path)
        root_path = self.voices_root or Path(self.settings.voices_dir)
        # Test configurations and a copied deploy config can retain canonical
        # /voices/... paths while mounting the private volume elsewhere.  Map
        # only that exact canonical root; arbitrary absolute paths remain
        # subject to the containment check below.
        canonical_root = Path("/voices")
        if candidate.is_absolute() and canonical_root in (candidate, *candidate.parents):
            try:
                candidate = root_path / candidate.relative_to(canonical_root)
            except ValueError:
                pass
        if not candidate.is_absolute():
            candidate = root_path / candidate
        try:
            root = root_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ReferenceValidationError("diretório de referências ausente") from exc
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ReferenceValidationError("arquivo de referência ausente") from exc
        except OSError as exc:
            raise ReferenceValidationError("caminho de referência inválido") from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ReferenceValidationError("caminho de referência fora do diretório permitido") from exc
        if not resolved.is_file():
            raise ReferenceValidationError("referência não é um arquivo regular")
        return resolved

    def _validate_audio(self, audio_path: Path, transcript_path: Path) -> ReferenceMetadata:
        try:
            info = sf.info(str(audio_path))
        except RuntimeError as exc:
            raise ReferenceValidationError("WAV de referência ilegível") from exc

        errors: list[str] = []
        warnings: list[str] = []
        codec = info.subtype or "unknown"
        duration = float(info.duration)
        if info.format != "WAV":
            errors.append("a referência deve ser um arquivo WAV")
        if not codec.upper().startswith("PCM"):
            errors.append("a referência deve usar codec PCM")
        if info.channels != 1:
            errors.append("a referência deve ser mono")
        if not 3.0 <= duration <= 60.0:
            errors.append("a duração da referência deve ficar entre 3 e 60 segundos")
        if info.samplerate <= 0:
            errors.append("sample rate inválido")

        try:
            waveform, _sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except RuntimeError as exc:
            raise ReferenceValidationError("não foi possível ler o áudio de referência") from exc
        if waveform.size == 0:
            errors.append("a referência não contém amostras")
            peak = rms = silence_ratio = clipping_ratio = 0.0
        else:
            absolute = np.abs(waveform)
            peak = float(np.max(absolute))
            rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
            silence_ratio = float(np.mean(absolute <= 0.001))
            clipping_ratio = float(np.mean(absolute >= 0.999))
            if clipping_ratio >= 0.001:
                warnings.append("clipping severo detectado")
            if silence_ratio >= 0.80:
                warnings.append("silêncio excessivo detectado")
        bits = _bits_from_subtype(codec)
        if info.samplerate != 32_000:
            warnings.append("sample rate diferente de 32.000 Hz; o original será enviado sem conversão")

        if errors:
            raise ReferenceValidationError("; ".join(errors))
        return ReferenceMetadata(
            duration_seconds=round(duration, 4),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            bits_per_sample=bits,
            codec=codec,
            size_bytes=audio_path.stat().st_size,
            peak=round(peak, 6),
            rms=round(rms, 6),
            silence_ratio=round(silence_ratio, 6),
            clipping_ratio=round(clipping_ratio, 6),
            audio_sha256=_sha256(audio_path),
            transcript_sha256=_sha256(transcript_path),
            warnings=tuple(warnings),
        )


def _bits_from_subtype(subtype: str) -> int | None:
    match = re.search(r"(?:PCM|FLOAT)_(\d+)", subtype.upper())
    return int(match.group(1)) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_error(error: Exception) -> str:
    """Reduce private filesystem failures to a stable operator-safe sentence."""

    return str(error).replace("\n", " ")[:300]
