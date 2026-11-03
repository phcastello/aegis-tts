"""Decision gating uses synthetic files; never open the held-out project corpus."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC=importlib.util.spec_from_file_location('aegis_v2_final_gate_test_module',Path(__file__).resolve().parents[1]/'benchmarks/v2/benchmark.py')
benchmark=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture(autouse=True)
def restore_process_umask():
    previous=os.umask(0o077);os.umask(previous)
    try:yield
    finally:os.umask(previous)


def decision():
    return dict(material_ttfa_gain_ms=400,max_wer_regression_pp=2,
        identity_rejection_rule='synthetic rule',selected_candidate='synthetic-candidate',validation_evidence='synthetic validation')


def arguments(tmp_path,path):
    return SimpleNamespace(endpoint='http://127.0.0.1:8001/v1/audio/speech',final_test=True,decision=str(path) if path else None,
        candidate='synthetic-candidate',case=None,output=str(tmp_path/'output'),sample_gpu=False,expected_pids=None,
        condition='warm',cold_evidence=None,payload='{"stream":true,"response_format":"pcm"}',text_field='input',
        seed_field=None,seed=40733133,repetitions=1,shuffle=False,timeout=1,stop_on_error=False)


@pytest.mark.parametrize('kind',['absent','missing_file','malformed','missing_keys','candidate_mismatch'])
def test_invalid_decision_keeps_final_corpus_closed(tmp_path,monkeypatch,kind):
    cohort=tmp_path/'synthetic-cohort';cohort.mkdir();final=cohort/'test.pt-BR.json'
    final.write_text('SENTINEL: this is deliberately not JSON',encoding='utf-8')
    path=tmp_path/'synthetic-decision.json'
    if kind=='absent':path=None
    elif kind=='malformed':path.write_text('{broken',encoding='utf-8')
    elif kind=='missing_keys':path.write_text(json.dumps(dict(selected_candidate='synthetic-candidate')),encoding='utf-8')
    elif kind=='candidate_mismatch':
        value=decision();value['selected_candidate']='different-candidate';path.write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(benchmark,'HERE',cohort)
    original=Path.read_text;opened=[]
    def guarded_read(self,*args,**kwargs):
        if self==final:
            opened.append(self);raise AssertionError('Final corpus opened before valid candidate-bound decision')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded_read)
    monkeypatch.setattr(benchmark,'private_output',lambda *args:pytest.fail('Output/benchmark began for invalid decision'))
    with pytest.raises((ValueError,FileNotFoundError)):
        benchmark.run(arguments(tmp_path,path))
    assert opened==[]


def test_valid_bound_decision_preserves_synthetic_run(tmp_path,monkeypatch):
    cohort=tmp_path/'synthetic-cohort';cohort.mkdir();final=cohort/'test.pt-BR.json'
    final.write_text(json.dumps([dict(id='synthetic_case',size='short',text='Uma frase sintética.')]),encoding='utf-8')
    path=tmp_path/'synthetic-decision.json';path.write_text(json.dumps(decision()),encoding='utf-8')
    monkeypatch.setattr(benchmark,'HERE',cohort)
    original=Path.read_text;opened=[]
    def tracked_read(self,*args,**kwargs):
        if self in (path,final):opened.append(self)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',tracked_read)
    pcm=b'\x10\x27'*960
    def receive(*_args):
        row=benchmark.pcm_metrics(pcm,[dict(ms=10.,bytes=len(pcm))])
        row.update(http_status=200,headers={},eof_ms=10.,rtf=.25,outcome='ok')
        return pcm,row
    monkeypatch.setattr(benchmark,'receive',receive)
    monkeypatch.setattr(benchmark.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='synthetic-commit'))
    benchmark.run(arguments(tmp_path,path))
    assert opened[:2]==[path,final]
    manifest=json.loads((tmp_path/'output/manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (tmp_path/'output/results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert manifest['corpus']==str(final) and manifest['decision_sha256']==benchmark.digest(path)
    assert len(rows)==1 and rows[0]['case']=='synthetic_case' and rows[0]['outcome']=='ok'
    assert (tmp_path/'output/synthetic_case-r00.wav').exists()
