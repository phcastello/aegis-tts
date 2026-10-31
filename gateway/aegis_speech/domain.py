"""Stable, transport-independent objects for the native speech API."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import monotonic


def now() -> datetime:
    return datetime.now(timezone.utc)


class RequestStatus(str, Enum):
    ACCEPTED = "accepted"
    NORMALIZING = "normalizing"
    QUEUED = "queued"
    GENERATING = "generating"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SegmentStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


FINAL_REQUEST_STATUSES = {RequestStatus.COMPLETED, RequestStatus.CANCELLED, RequestStatus.FAILED}
_TRANSITIONS = {
    RequestStatus.ACCEPTED: {RequestStatus.NORMALIZING, RequestStatus.CANCELLING, RequestStatus.FAILED},
    RequestStatus.NORMALIZING: {RequestStatus.QUEUED, RequestStatus.CANCELLING, RequestStatus.FAILED},
    RequestStatus.QUEUED: {RequestStatus.GENERATING, RequestStatus.CANCELLING, RequestStatus.FAILED},
    RequestStatus.GENERATING: {RequestStatus.STREAMING, RequestStatus.QUEUED, RequestStatus.COMPLETED, RequestStatus.CANCELLING, RequestStatus.FAILED},
    RequestStatus.STREAMING: {RequestStatus.GENERATING, RequestStatus.QUEUED, RequestStatus.COMPLETED, RequestStatus.CANCELLING, RequestStatus.FAILED},
    RequestStatus.CANCELLING: {RequestStatus.CANCELLED, RequestStatus.FAILED},
    RequestStatus.COMPLETED: set(), RequestStatus.CANCELLED: set(), RequestStatus.FAILED: set(),
}


@dataclass
class SpeechRequest:
    request_id: str
    original_text: str
    normalized_text: str | None
    priority: int
    interrupt_policy: str
    voice_profile: str
    stream: bool
    response_format: str
    metadata: dict[str, str]
    status: RequestStatus = RequestStatus.ACCEPTED
    created_at: datetime = field(default_factory=now)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    first_audio_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    error_code: str | None = None
    cancel_reason: str | None = None
    created_monotonic: float = field(default_factory=monotonic)

    def transition(self, status: RequestStatus) -> None:
        if status == self.status:
            return
        if status not in _TRANSITIONS[self.status]:
            raise ValueError(f"invalid request transition: {self.status.value} -> {status.value}")
        self.status = status
        stamp = now()
        if status == RequestStatus.QUEUED:
            self.queued_at = stamp
        elif status == RequestStatus.GENERATING:
            self.started_at = self.started_at or stamp
        elif status == RequestStatus.COMPLETED:
            self.completed_at = stamp
        elif status == RequestStatus.CANCELLED:
            self.cancelled_at = stamp


@dataclass
class SpeechSegment:
    request_id: str
    segment_index: int
    original_text: str
    normalized_text: str
    status: SegmentStatus
    estimated_cost: float
    created_at: datetime = field(default_factory=now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    original_span: tuple[int, int] = (0, 0)
    normalized_span: tuple[int, int] = (0, 0)
    split_reason: str = "final_fragment"
