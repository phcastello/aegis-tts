import json, os, subprocess, sys
from pathlib import Path
import pytest

@pytest.mark.graillon_integration
def test_real_graillon_is_opt_in(tmp_path):
    if os.getenv('AEGIS_GRAILLON_INTEGRATION_TESTS')!='1':
        pytest.skip('set AEGIS_GRAILLON_INTEGRATION_TESTS=1')
    # The CLI is the integration contract.  Separate processes prove the
    # per-preset reset policy, then compare the raw outputs.
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'results'
    outputs={}
    for preset in ('G00','G01'):
        p=subprocess.run([sys.executable,'tools/graillon_lab.py','render','--preset',preset,'--output-dir',str(output)],cwd=root,text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        manifest=json.loads((output/'manifest.json').read_text())
        record=manifest['presets'][0]
        raw=output/'raw'/record['raw_file']
        assert record['success'] and raw.stat().st_size>44
        outputs[preset]=raw.read_bytes()
    assert outputs['G00'] != outputs['G01']
    ps=subprocess.run(['ps','-eo','args='],text=True,capture_output=True,check=True).stdout
    assert 'jackd --no-realtime -n aegis-glados-lab' not in ps
    assert 'jalv -i -p -x -n aegis-graillon-plugin' not in ps
