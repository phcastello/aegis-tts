import json, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import glados_lab as lab
import glados_stream_test as stream

def presets(): return lab.load_yaml(ROOT/'config/glados-presets.yaml')

def test_yaml_valid_and_unique_ids():
    data=presets(); assert len(data)==8
    assert len({x['id'] for x in data})==len(data)
    assert all(x['streaming_safe'] and x['filtergraph'] for x in data)

def test_pitch_metadata_is_consistent():
    for p in presets():
        q=p['parameters']
        if 'pitch_factor' in q:
            assert q['pitch_factor'] == pytest.approx(2**(q['pitch_semitones']/12),abs=2e-5)
            assert q.get('sample_rate_final',24000)==24000

def test_interpolation_and_filter_detection():
    p=next(x for x in presets() if x['id']=='P04')
    graph=lab.interpolate(p['filtergraph'],p['parameters'])
    assert '{' not in graph and lab.needed_filters(graph)=={'vibrato'}

def test_commands_are_argument_lists_and_paths_are_not_shell_expanded(tmp_path, monkeypatch):
    seen=[]
    class Result: returncode=0; stderr=''; stdout=''
    monkeypatch.setattr(lab,'command',lambda args,**kw: seen.append(args) or Result())
    lab.render(tmp_path/'odd;name.wav',tmp_path/'out.wav','anull')
    assert seen[0][0]=='ffmpeg' and 'shell' not in str(seen[0]).lower()
    assert str(tmp_path/'odd;name.wav') in seen[0]

def test_one_failed_preset_does_not_prevent_manifest_html(tmp_path):
    manifest={'source_sha256':'abc','presets':[{'id':'A','name':'a','family':'control','description':'x','parameters':{},'file':'no.wav','success':False},{'id':'B','name':'b','family':'control','description':'x','parameters':{},'file':'yes.wav','success':True}]}
    lab.write_outputs(tmp_path,manifest)
    assert (tmp_path/'manifest.json').exists() and (tmp_path/'listening.html').exists()
    assert json.loads((tmp_path/'blind-map.json').read_text())['Sample 01']=='B'

def test_html_references_successful_files(tmp_path):
    manifest={'source_sha256':'x','presets':[{'id':'A','name':'a','family':'control','description':'x','parameters':{},'file':'preset-A.wav','success':True}]}
    lab.write_outputs(tmp_path,manifest)
    assert 'preset-A.wav' in (tmp_path/'listening.html').read_text()

def test_stream_command_uses_stdin_stdout_without_shell(monkeypatch):
    captured={}
    class Pipe:
        def read(self,*a): return b''
        def write(self,*a): return 0
        def flush(self): pass
        def close(self): pass
    class Proc:
        stdin=Pipe(); stdout=Pipe(); stderr=Pipe()
        def wait(self,timeout=None): return 0
        def terminate(self): pass
    monkeypatch.setattr(stream.subprocess,'Popen',lambda cmd,**kwargs: captured.update(cmd=cmd,kwargs=kwargs) or Proc())
    result,_=stream.run_case(b'\0'*16,'anull',4096,0)
    assert captured['cmd'][-1]=='pipe:1' and result['exit_code']==0
    assert captured['kwargs'].get('shell',False) is False
