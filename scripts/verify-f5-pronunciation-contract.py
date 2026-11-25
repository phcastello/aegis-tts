#!/usr/bin/env python3
"""Exercise the deployed F5 HTTP adapter with a fake CPU engine, not production.

The F5 service is installed as a separate versioned runtime. Point this check
at its source directory (or a verified copy), and at the candidate lexicon.
Neither the speech model nor the real service is started or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = "Aegis abriu o Gmail. A GPU, a VPN, o DNS e o Qwen estão disponíveis."
SPOKEN = "Aéjis abriu o gê mail. A gê pê u, a vê pê ene, o dê ene ésse e o Cuén estão disponíveis."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, default=ROOT / "config/pronunciation/pt-BR-f5-v2.yaml")
    parser.add_argument("--expected-runtime-sha256")
    args = parser.parse_args()

    runtime = args.runtime_dir.resolve(strict=True)
    lexicon = args.lexicon.resolve(strict=True)
    app_hash = hashlib.sha256((runtime / "app.py").read_bytes()).hexdigest()
    if args.expected_runtime_sha256 and app_hash != args.expected_runtime_sha256:
        raise SystemExit("F5 runtime does not match the expected app.py hash")

    # Resolve imports from the actual F5 gateway, not a reimplementation.
    sys.path.insert(0, str(runtime))
    import numpy as np
    from fastapi.testclient import TestClient
    from app import create_app, NATIVE_PROFILE

    class FakeEngine:
        max_context_frames = 8192
        whole_utterance = False

        def __init__(self, _manifest):
            self.calls = []

        def required_frames(self, text):
            return 100 + len(text)

        def validate_text(self, text):
            return {"total_context_frames": self.required_frames(text)}

        def synthesize_stream(self, text, _stop):
            self.calls.append(text)
            # This artificial waveform is used only for exercising transport.
            for _ in range(2):
                yield np.zeros(2400, dtype=np.float32), {"whole_utterance": False}

    app = create_app(engine_factory=FakeEngine, lexicon_path=lexicon, prewarm=False)
    with TestClient(app) as client:
        native = client.post("/v1/aegis/speech", json={
            "text": TEXT, "stream": True, "response_format": "pcm", "voice_profile": NATIVE_PROFILE,
        })
        assert native.status_code == 200
        assert native.headers["x-aegis-audio-format"] == "pcm_s16le"
        assert len(native.content) == 9600
        manual = client.post("/v1/audio/speech", json={"input": TEXT, "response_format": "wav"})
        assert manual.status_code == 200
        with wave.open(io.BytesIO(manual.content), "rb") as wav:
            assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 24000, 4800)
        assert app.state.engine.calls == [SPOKEN, SPOKEN]

        # The original text survives the normalization boundary. Only what the
        # TTS engine receives contains the phonetic aliases.
        for response in (native, manual):
            request_id = response.headers["x-aegis-request-id"]
            speech = app.state.service.runtimes[request_id].request
            assert speech.original_text == TEXT
            assert speech.normalized_text == SPOKEN

    print(f"F5 pronunciation contract passed: native PCM + manual WAV; runtime app.py SHA-256={app_hash}")


if __name__ == "__main__":
    main()
