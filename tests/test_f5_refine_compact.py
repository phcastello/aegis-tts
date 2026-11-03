"""Compact delivery copies one representation while preserving native evidence."""
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

from training import f5_listening_compat as compat
from training import f5_refine_compact as compact


def full_package(root):
    source=root/'listening';source.mkdir(mode=0o700,parents=True)
    manifest={};metrics=[];parts=['<h1>Escuta</h1><a href="report.json">Relatório</a>']
    for i,(name,values,subtype) in enumerate([
        ('safe',[0,.1234567,-.25],'FLOAT'),
        ('overrange',[0,1.2,-.25],'FLOAT'),
        ('pcm',[0,.1,-.2],'PCM_16'),
        ('reference',[0,.3,-.4],'PCM_24'),
    ]):
        relative='audio/'+name+'.wav';path=source/relative;path.parent.mkdir(exist_ok=True)
        sf.write(path,np.array(values),24000,subtype=subtype)
        manifest[relative]=compat.digest(path)
        parts.append('<audio controls src="'+relative+'"></audio>')
        if name!='reference':metrics.append(dict(id=name,stage='controlled',case=name,label=name,
            native_path=relative,sha256=manifest[relative],peak=max(abs(x) for x in values),
            display_text='Exibido',tts_text='Fala',asr_expected='Fala'))
    (source/'index.html').write_text('\n'.join(parts))
    (source/'manifest-sha256.json').write_text(json.dumps(manifest))
    (source/'metrics.json').write_text(json.dumps(metrics))
    (source/'metrics.csv').write_text('id,native_path,sha256,asr_expected\n')
    (source/'scorecard.csv').write_text('id,score\nsafe,\n')
    (source/'report.json').write_text(json.dumps({'owner_approval':False}))
    compat.process(source)
    artifacts={p.relative_to(source).as_posix():compact.digest(p) for p in source.rglob('*') if p.is_file()}
    (source/'artifacts-sha256.json').write_text(json.dumps(artifacts))
    return source


def test_compact_one_wav_per_clip_with_truthful_hashes_and_unchanged_sources(tmp_path):
    source=full_package(tmp_path)
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=compact.build(tmp_path)
    assert result['compact_wav_files']==4  # Includes reference absent from metrics.
    assert result['pcm16_selected']==3 and result['float_originals_retained']==1
    assert all(p.read_bytes()==data for p,data in before.items())
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.testzip() is None
        wavs=[n for n in archive.namelist() if n.endswith('.wav')]
        assert len(wavs)==4
        assert not any(n.startswith('playback/') for n in wavs)
        manifest=json.loads(archive.read('manifest-sha256.json'))
        assert all(compact.hashlib.sha256(archive.read(n)).hexdigest()==sha for n,sha in manifest.items())
        receipt=json.loads(archive.read('compact-receipt.json'))
        by_name={Path(r['packaged_path']).stem:r for r in receipt['files']}
        overrange=by_name['overrange'];safe=by_name['safe']
        assert archive.read(overrange['packaged_path'])==before[source/overrange['original_native_path']]
        assert overrange['packaged_sha256']==overrange['original_native_sha256']
        assert safe['packaged_sha256']!=safe['original_native_sha256']
        metrics=json.loads(archive.read('metrics.json'))
        safe_metric=next(r for r in metrics if r['id']=='safe')
        assert safe_metric['sha256']==safe_metric['original_native_sha256']
        assert safe_metric['packaged_sha256']==safe['packaged_sha256']
        assert 'native_path' not in safe_metric
        html=archive.read('index.html').decode()
        assert 'data-native-download' not in html
        assert 'href="report.json"' in html
        assert 'Os nativos completos continuam' in html
        compact.verify_directory(tmp_path/'listening-compact')
