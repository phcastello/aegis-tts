from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))

from aegis_speech.domain import RequestStatus, SegmentStatus, SpeechRequest, SpeechSegment
from aegis_speech.lexicon import LexiconError, PronunciationLexicon
from aegis_speech.normalization import SpeechTextNormalizer
from aegis_speech.segmentation import SpeechSegmenter
from aegis_speech.service import BusyError, InterruptionPlanner, QueueFullError, SpeechService


def normalizer() -> SpeechTextNormalizer:
    return SpeechTextNormalizer(PronunciationLexicon(ROOT / "config/pronunciation/pt-BR.yaml").apply)


@pytest.mark.parametrize(("source", "expected"), [
    ("A GPU chegou a 62%.", "A gê pê u chegou a sessenta e dois por cento."),
    ("O servidor 10.1.1.47 responde na porta 8001.", "O servidor dez ponto um ponto um ponto quarenta e sete responde na porta oito mil e um."),
    ("A versão 1.4.11 está ativa.", "A versão um ponto quatro ponto onze está ativa."),
    ("Agora são 14:35.", "Agora são quatorze horas e trinta e cinco minutos."),
    ("A temperatura é de -3,5 °C.", "A temperatura é de menos três vírgula cinco graus Celsius."),
    ("O uso chegou a 3,5 GB.", "O uso chegou a três vírgula cinco gigabytes."),
])
def test_required_normalization_cases(source: str, expected: str) -> None:
    assert normalizer().normalize(source).normalized_text == expected


def test_normalization_is_idempotent_for_spoken_output_and_removes_bidi() -> None:
    first = normalizer().normalize("A GPU\u202e chegou a 62%.").normalized_text
    assert "\u202e" not in first
    assert normalizer().normalize(first).normalized_text == first


def test_lexicon_has_token_boundaries_and_rejects_self_cycle(tmp_path: Path) -> None:
    lexicon = PronunciationLexicon(ROOT / "config/pronunciation/pt-BR.yaml")
    assert lexicon.apply("detecção GPUx GPU")[0].endswith("GPUx gê pê u")
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\nentries:\n- id: bad\n  match: {type: token, value: A}\n  replacement: A\n")
    with pytest.raises(LexiconError):
        PronunciationLexicon(path)


def test_segmenter_does_not_emit_empty_segments_or_split_technical_tokens() -> None:
    result = normalizer().normalize("O IP é 10.1.1.47. A versão 1.4.11 está ativa. Segunda frase.")
    drafts = SpeechSegmenter(min_chars=1).segment(result)
    assert all(item.normalized_text for item in drafts)
    assert [item.segment_index for item in drafts] == list(range(len(drafts)))


def test_state_machine_rejects_invalid_transition() -> None:
    request = SpeechRequest("0" * 26, "x", None, 50, "enqueue", "AegisVoicev1.0", True, "pcm", {})
    with pytest.raises(ValueError):
        request.transition(RequestStatus.COMPLETED)


async def _adapter(segment: SpeechSegment, cancelled: asyncio.Event):
    for _ in range(2):
        if cancelled.is_set(): return
        yield b"\x00\x00"
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_scheduler_priority_and_cancellation_are_bounded() -> None:
    service = SpeechService(_adapter, max_requests=3)
    await service.start()
    def make(identifier: str, priority: int) -> tuple[SpeechRequest, list[SpeechSegment]]:
        request = SpeechRequest(identifier, "texto suficiente", "texto suficiente", priority, "enqueue", "AegisVoicev1.0", True, "pcm", {})
        return request, [SpeechSegment(identifier, 0, "texto suficiente", "texto suficiente", SegmentStatus.QUEUED, 1)]
    low, low_segments = make("0" * 26, 10)
    high, high_segments = make("1" * 26, 90)
    low_runtime = await service.submit(low, low_segments)
    high_runtime = await service.submit(high, high_segments)
    await service.cancel(low_runtime.request.request_id)
    await asyncio.wait_for(high_runtime.done.wait(), 2)
    assert high.status == RequestStatus.COMPLETED
    assert low.status == RequestStatus.CANCELLED
    await service.stop()


@pytest.mark.asyncio
async def test_reject_if_busy() -> None:
    service = SpeechService(_adapter)
    await service.start()
    first = SpeechRequest("2" * 26, "x", "x", 50, "enqueue", "AegisVoicev1.0", True, "pcm", {})
    await service.submit(first, [SpeechSegment(first.request_id, 0, "x", "x", SegmentStatus.QUEUED, 1)])
    second = SpeechRequest("3" * 26, "x", "x", 50, "reject_if_busy", "AegisVoicev1.0", True, "pcm", {})
    with pytest.raises(BusyError):
        await service.submit(second, [SpeechSegment(second.request_id, 0, "x", "x", SegmentStatus.QUEUED, 1)])
    await service.stop()


def _request(identifier: str, priority: int = 50, policy: str = "enqueue") -> SpeechRequest:
    return SpeechRequest(identifier * 26, "texto suficiente", "texto suficiente", priority, policy, "AegisVoicev1.0", True, "pcm", {})


def _segment(request: SpeechRequest) -> SpeechSegment:
    return SpeechSegment(request.request_id, 0, request.original_text, request.normalized_text or request.original_text, SegmentStatus.QUEUED, 1)


def test_interruption_policy_only_replaces_lower_priority_and_gates_replace_all() -> None:
    active = _request("a", priority=30)
    equal_priority = _request("b", priority=80)
    incoming = _request("c", priority=80, policy="replace_lower_priority")
    cancelled, preserved = InterruptionPlanner().apply_policy(incoming, active, [equal_priority])
    assert cancelled == [active.request_id]
    assert preserved == [equal_priority.request_id]
    with pytest.raises(PermissionError):
        InterruptionPlanner().apply_policy(_request("d", policy="replace_all"), active, [equal_priority])
    cancelled, preserved = InterruptionPlanner(allow_replace_all=True).apply_policy(_request("e", policy="replace_all"), active, [equal_priority])
    assert cancelled == [active.request_id, equal_priority.request_id]
    assert preserved == []


@pytest.mark.asyncio
async def test_same_priority_is_fifo_and_events_are_monotonic() -> None:
    service = SpeechService(_adapter)
    first, second = _request("f"), _request("g")
    first_runtime = await service.submit(first, [_segment(first)])
    await service.submit(second, [_segment(second)])
    assert service.queue_position(first.request_id) == 1
    assert service.queue_position(second.request_id) == 2
    assert [event.sequence for event in first_runtime.events] == [1, 2, 3, 4]
    assert [event.event for event in first_runtime.events] == ["accepted", "normalization_completed", "queued", "segment_queued"]
    await service.stop()


@pytest.mark.asyncio
async def test_queue_capacity_and_pending_cancellation_are_enforced() -> None:
    service = SpeechService(_adapter, max_requests=1)
    first = _request("h")
    runtime = await service.submit(first, [_segment(first)])
    with pytest.raises(QueueFullError):
        second = _request("i")
        await service.submit(second, [_segment(second)])
    await service.cancel(first.request_id)
    assert runtime.request.status == RequestStatus.CANCELLED
    assert runtime.request.cancel_reason == "client_request"
    assert runtime.events[-1].event == "cancelled"
    await service.stop()
