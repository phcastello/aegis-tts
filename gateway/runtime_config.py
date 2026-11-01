"""Derive a private codec cadence config without editing protected inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def prepare(source: Path, output: Path, codec_frames: int) -> Path:
    if codec_frames not in {16, 22, 25}:
        raise ValueError('unsupported codec cadence; validated experiments are 16, 22 and 25')
    config = yaml.safe_load(source.read_text())
    extra = config['connectors']['connector_of_shared_memory']['extra']
    if not isinstance(extra.get('codec_chunk_frames'), int):
        raise ValueError('missing effective codec batch size')
    if extra['codec_chunk_frames'] == codec_frames:
        return source
    if source.resolve() == output.resolve():
        raise ValueError('protected source must never be overwritten')
    # Every other key, including sampling, reference context, graph shapes and
    # resource limits, is preserved. The per-request initial override is owned
    # by the gateway; configured initial=1 remains unchanged here.
    if extra.get('initial_codec_chunk_frames', 0) > codec_frames:
        raise ValueError('configured initial frames exceed new codec cadence')
    extra['codec_chunk_frames'] = codec_frames
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        os.fchmod(handle.fileno(), 0o600)
        yaml.safe_dump(config, handle, sort_keys=False)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codec-frames', type=int, required=True)
    args = parser.parse_args()
    print(prepare(args.input, args.output, args.codec_frames))
