"""CPU harness safety checks; no model, Torch, CUDA or network required."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/architecture_compare/magpie_codec_probe.py"
SPEC = importlib.util.spec_from_file_location("magpie_codec_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_plan_is_dependency_free(capsys):
    assert probe.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan"
    assert plan["cuda"] is plan["network"] is plan["training"] is False


def test_changed_upstream_bytes_rejected_before_deserialization(tmp_path):
    for name in probe.HASHES:
        (tmp_path / name).write_bytes(b"not the verified upstream")
    with pytest.raises(ValueError, match="unverified official artifact"):
        probe.verified_files(tmp_path)


def test_numeric_expressions_and_non_metadata_decorators_preserved():
    original = ast.parse("""
class Encoder:
    @property
    def factor(self):
        return 1024
    @typecheck(input_types={'x': NeuralType()})
    @torch.jit.script
    def forward(self, x):
        return (x + 0.5).tanh() * 7
""")
    numeric_body = ast.dump(original.body[0].body[1].body[0])
    transformed = probe.MetadataOnly().visit(original)
    assert ast.dump(transformed.body[0].body[1].body[0]) == numeric_body
    assert transformed.body[0].body[1].decorator_list == []
    assert ast.unparse(transformed.body[0].body[0].decorator_list[0]) == "property"
