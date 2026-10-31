"""Cached reference serialization must preserve the complete upstream contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
import default_reference_gateway as gateway


@pytest.mark.parametrize('text', [
    'São nove horas e quinze minutos.',
    'Água, açúcar e café.\nDuas linhas; emoji: 🌊',
    '\"},\"ref_audio\":\"injected\",\"temperature\":99,\"input\":\"',
])
@pytest.mark.parametrize('cached', [False, True])
def test_cached_and_fallback_requests_preserve_exact_payload(monkeypatch, text, cached):
    monkeypatch.setattr(gateway.app.state, 'default_audio', 'data:audio/wav;base64,synthetic', raising=False)
    monkeypatch.setattr(gateway.app.state, 'default_text', 'Synthetic \"reference\".\nSecond line.', raising=False)
    expected = gateway._apply_defaults({'input': text, 'stream': True, 'response_format': 'pcm'})
    prefix = gateway._segment_request_prefix() if cached else None
    monkeypatch.setattr(gateway.app.state, 'segment_request_prefix', prefix, raising=False)
    with httpx.Client(base_url='http://example.test') as client:
        monkeypatch.setattr(gateway.app.state, 'client', client, raising=False)
        request = gateway._segment_upstream_request(text)
    assert request.method == 'POST'
    assert request.url.path == '/v1/audio/speech'
    assert request.headers['content-type'] == 'application/json'
    assert int(request.headers['content-length']) == len(request.content)
    assert json.loads(request.content) == expected
    assert json.loads(request.content)['input'] == text
    if cached:
        assert prefix.endswith(b',"input":')
        assert b'"input"' not in prefix[:-9]
