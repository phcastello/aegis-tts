from copy import deepcopy
import os
from pathlib import Path
import sys
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from runtime_config import prepare


def test_codec_cadence_changes_only_one_setting_and_preserves_source(tmp_path):
    original = {'connectors': {'connector_of_shared_memory': {'extra': {
        'codec_chunk_frames': 25, 'initial_codec_chunk_frames': 1,
        'codec_left_context_frames': 72, 'decode_cudagraph_capture_sizes': [25, 73]}}},
        'stages': [{'max_num_seqs': 1, 'default_sampling_params': {'seed': 40733133}}]}
    source, output = tmp_path / 'protected.yaml', tmp_path / 'runtime.yaml'
    source.write_text(yaml.safe_dump(original)); before = source.read_bytes()
    assert prepare(source, output, 25) == source
    assert not output.exists()
    assert prepare(source, output, 22) == output
    expected = deepcopy(original)
    expected['connectors']['connector_of_shared_memory']['extra']['codec_chunk_frames'] = 22
    assert yaml.safe_load(output.read_text()) == expected
    assert source.read_bytes() == before
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match='overwritten'):
        prepare(source, source, 22)
    with pytest.raises(ValueError, match='unsupported'):
        prepare(source, output, 12)
