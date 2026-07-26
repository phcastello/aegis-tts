"""Apply the smallest reproducible diagnostic fix from vLLM-Omni PR #1945.

This file is used only by Dockerfile.code2wav-diagnostic.  It intentionally
changes neither Stage 0, ICL reference, sampling, nor the gateway DSP.
"""

from pathlib import Path


target = Path("/usr/local/lib/python3.12/dist-packages/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py")
source = target.read_text(encoding="utf-8")
old = """            # Slice on exact codec-frame boundaries instead of proportionally.
            start = max(0, ctx_frames * upsample)
            end = max(start, actual_frames * upsample)
            if start >= wav.shape[0]:
                logger.warning(
                    \"Context trim start %d >= decoded length %d; returning empty audio.\",
                    start,
                    wav.shape[0],
                )
                continue
            wav = wav[start : min(end, wav.shape[0])]
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)
"""
new = """            decoded_wav_samples = int(wav.shape[0])
            # PR #1945: decoder length varies slightly, so trim the context by
            # its proportion of the actual decoded frame window rather than a
            # fixed ctx_frames * upsample count.
            context_trim_samples = int(ctx_frames / max(actual_frames, 1) * decoded_wav_samples)
            if context_trim_samples >= decoded_wav_samples:
                logger.warning(
                    \"AEGIS_CODE2WAV_TRIM chunk=%d ctx_frames=%d actual_frames=%d decoded_wav_samples=%d context_trim_samples=%d emitted_wav_samples=0\",
                    idx, ctx_frames, actual_frames, decoded_wav_samples, context_trim_samples,
                )
                continue
            wav = wav[context_trim_samples:]
            logger.warning(
                \"AEGIS_CODE2WAV_TRIM chunk=%d ctx_frames=%d actual_frames=%d decoded_wav_samples=%d context_trim_samples=%d emitted_wav_samples=%d\",
                idx, ctx_frames, actual_frames, decoded_wav_samples, context_trim_samples, int(wav.shape[0]),
            )
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)
"""
if old not in source:
    raise SystemExit("expected vLLM-Omni 0.22.0 fixed-trim block was not found")
target.write_text(source.replace(old, new, 1), encoding="utf-8")
