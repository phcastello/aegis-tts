"""Freeze a VITS-specific public-corpus selection without changing FP inputs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from training.f5_refine_text import normalize_text
from training.v2_vits_compat_audit import frontend

QUARANTINE = {'sample-72','sample-119','sample-248','sample-348','sample-356','sample-395',
              'sample-1634','sample-2459','sample-2966','sample-4724','sample-1430',
              'sample-47','sample-70','sample-2142','sample-2252'}
PUNCTUATION = {'\u201c':'"','\u201d':'"','\u2018':"'",'\u2019':"'",'\u2013':'-',
               '\u2014':'-','\u2212':'-','\u00a0':' ','\u2026':'...'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/var/tmp/aegis-v2-work/portuguese/prepared'))
    p.add_argument('--output',type=Path,default=Path('/var/tmp/aegis-v2-work/portuguese/vits-prepared'))
    p.add_argument('--vits-root',type=Path,default=Path('/var/tmp/aegis-v2-work/vits'))
    a=p.parse_args();os.umask(0o077)
    source_hashes={s:sha(a.source/(s+'.jsonl')) for s in ('train','validation','test')}
    config=json.loads((a.vits_root/'model/tts_models--pt--cv--vits/config.json').read_text())
    tokenizer=frontend(a.vits_root/'deps/TTS/tts/utils/text',config)
    source_rows={s:[json.loads(x) for x in (a.source/(s+'.jsonl')).read_text().splitlines()] for s in source_hashes}
    selected,quarantine,changes={s:[] for s in source_hashes},[],[]
    seen_ids=set();split_keys={k:defaultdict(set) for k in ('group','text','sha')}
    for split,rows in source_rows.items():
        for row in rows:
            if row['id'] in QUARANTINE:
                quarantine.append(dict(id=row['id'],split=split,reason='explicit initial frontend ambiguity quarantine pending reviewed transcription; no guessed alphanumeric pronunciation'))
                continue
            effective=unicodedata.normalize('NFC',row['text']).translate(str.maketrans(PUNCTUATION))
            effective=' '.join(effective.split())
            tokenizer.not_found_characters.clear();ids=tokenizer.text_to_ids(normalize_text(effective)['tts_text'],language='pt')
            if tokenizer.not_found_characters or not ids:
                quarantine.append(dict(id=row['id'],split=split,reason='additional unsupported grapheme',characters=tokenizer.not_found_characters.copy()))
                continue
            if row['id'] in seen_ids:raise ValueError('remaining duplicate source ID: '+row['id'])
            seen_ids.add(row['id'])
            for kind,key in (('group',row['group_id']),('text',' '.join(re.findall(r'\w+',effective.casefold()))),('sha',row['sha256'])):
                split_keys[kind][key].add(split)
            if effective != row['text']:
                changes.append(dict(id=row['id'],split=split,before=row['text'],after=effective,policy='NFC, explicit Unicode punctuation map and whitespace only'))
            selected[split].append({**row,'text':effective,'source_prepared_text':row['text'],
                                    'source_prepared_manifest_sha256':source_hashes[split],
                                    'transcription_policy':'text is reviewed effective transcription; original_text preserved as source provenance'})
    overlaps={k:sum(len(v)>1 for v in mapping.values()) for k,mapping in split_keys.items()}
    if any(overlaps.values()):raise ValueError('cross-split leakage: '+repr(overlaps))
    if source_hashes!={s:sha(a.source/(s+'.jsonl')) for s in source_hashes}:raise ValueError('source manifests changed during freeze')
    staging=a.output.with_name(a.output.name+'.staging');staging.mkdir(mode=0o700,parents=True,exist_ok=False)
    for split,rows in selected.items():
        (staging/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(schema='aegis.vits.public-selection.v1',source=str(a.source),source_hashes=source_hashes,
                selected_counts={s:len(rows) for s,rows in selected.items()},
                selected_seconds={s:sum(r['duration'] for r in rows) for s,rows in selected.items()},
                selected_manifest_hashes={s:sha(staging/(s+'.jsonl')) for s in selected},
                quarantine=quarantine,quarantine_unique_ids=len({r['id'] for r in quarantine}),
                source_duplicate_quarantined_ids=dict(Counter(r['id'] for r in quarantine)),
                punctuation_changes=changes,punctuation_map=PUNCTUATION,leakage=overlaps,
                original_text_policy='Preserved byte-for-byte in each JSON value; text is reviewed transcript used for training',
                audio_policy='No copies/conversion. Hash-bound existing24k prepared WAVs; runtime verifies hash and resamples once to22050',
                code_sha256=sha(__file__))
    (staging/'selection-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for file in staging.iterdir():file.chmod(0o400)
    if a.output.exists():raise FileExistsError('refusing to replace frozen selection')
    staging.rename(a.output)
    print(json.dumps({k:report[k] for k in ('selected_counts','selected_seconds','quarantine_unique_ids','leakage','selected_manifest_hashes')}))


if __name__=='__main__':main()
