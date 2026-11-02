#!/usr/bin/env python3
"""Private, hash-bound ZIP audit and F5/Vocos CPU preparation (no effects)."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import unicodedata
import zipfile

EXPECTED = '5bbd8288d9ab795a83ffd1599b0c5a32c00b401cb903b368a183d7ee7580c8df'
RATE = 24000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    path.chmod(0o600)


def records(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def safe_member(name):
    p = PurePosixPath(name)
    if '\\' in name or '\x00' in name or p.is_absolute() or '..' in p.parts or not p.parts or ':' in p.parts[0]:
        raise ValueError(f'unsafe ZIP member: {name!r}')
    return p


def extract(archive, destination, expected=EXPECTED, max_bytes=3_000_000_000):
    """Validate whole archive and every CRC before publishing a private extraction."""
    archive = Path(archive)
    actual = digest(archive)
    if actual != expected:
        raise ValueError(f'ZIP SHA256 mismatch: {actual}')
    destination = Path(destination)
    if destination.exists():
        receipt = destination / 'extraction-receipt.json'
        if not receipt.exists() or json.loads(receipt.read_text())['archive_sha256'] != actual:
            raise ValueError('existing extraction is not the same verified archive')
        return json.loads(receipt.read_text())
    staging = destination.with_name(destination.name+'.staging')
    staging.mkdir(mode=0o700, parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = set()
        total = 0
        for item in z.infolist():
            p = safe_member(item.filename)
            if str(p) in names:
                raise ValueError('duplicate ZIP destination')
            names.add(str(p))
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('nonregular ZIP entry')
            total += item.file_size
            if item.flag_bits & 1 or total > max_bytes or item.file_size > 100_000_000:
                raise ValueError('encrypted or oversized ZIP entry')
        if shutil.disk_usage(staging).free < total + 6_000_000_000:
            raise ValueError('insufficient free space with operational reserve')
        for item in z.infolist():
            target = staging.joinpath(*safe_member(item.filename).parts)
            if item.is_dir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            count = 0
            with z.open(item) as src, target.open('xb') as dst:
                while block := src.read(1024*1024):
                    count += len(block)
                    if count > item.file_size:
                        raise ValueError('ZIP member expanded beyond advertised size')
                    dst.write(block)
            if count != item.file_size:
                raise ValueError('truncated ZIP member')
            target.chmod(0o600)
    receipt = {'archive_sha256': actual, 'archive_bytes': archive.stat().st_size,
               'members': len(names), 'uncompressed_bytes': total, 'crc_all_verified': True,
               'path_validation': 'relative POSIX paths; no traversal, symlinks, devices, encryption or duplicate entries'}
    dump(staging/'extraction-receipt.json', receipt)
    staging.rename(destination)
    return receipt


def canonical_text(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))


def verify_prepared(prepared):
    """Recheck generated manifests and every source/WAV/mel artifact on reuse."""
    import numpy as np
    prepared = Path(prepared)
    audit = json.loads((prepared/'audit.json').read_text())
    count = 0
    for split in ('train', 'validation', 'test'):
        manifest = prepared/f'{split}.jsonl'
        if digest(manifest) != audit['manifest_hashes'][split]:
            raise ValueError('prepared manifest changed')
        for row in records(manifest):
            for path_key, hash_key in (('source_audio_filepath', 'source_audio_sha256'),
                                       ('audio_filepath', 'sha256'), ('mel_filepath', 'mel_sha256')):
                if digest(row[path_key]) != row[hash_key]:
                    raise ValueError(f'{row["id"]}: cached artifact changed: {path_key}')
            mel = np.load(row['mel_filepath'], allow_pickle=False)
            if list(mel.shape) != row['mel_shape'] or not np.isfinite(mel).all():
                raise ValueError('invalid cached mel')
            count += 1
    if count != audit['accepted_count']:
        raise ValueError('prepared count changed')
    return {'artifacts_reverified': count, 'source_manifest_sha256': audit['source_manifest_sha256']}


def prepare(args):
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    root = args.root.resolve()
    repository = Path(__file__).resolve().parents[1]
    if root == repository or repository in root.parents:
        raise ValueError('private root must be outside repository')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    started = time.monotonic()
    receipt = extract(args.zip, root/'extracted')
    print(json.dumps({'stage':'extracted',**receipt}),flush=True)
    source = root/'extracted/dataset_final_124_7ppm'
    rows = records(source/'dataset.jsonl')
    prepared = root/'prepared'
    if (prepared/'audit.json').exists():
        print(json.dumps(verify_prepared(prepared)))
        print((prepared/'audit.json').read_text())
        return
    prepared.mkdir(mode=0o700, exist_ok=True)
    for name in ('audio','mels','splits'):
        (prepared/name).mkdir(mode=0o700,exist_ok=True)
    import numpy as np
    import soundfile as sf
    import torch
    from training.prepare_dataset import read_pcm, resample_pcm
    sys.path.insert(0,str(args.f5_source/'src'))
    from f5_tts.model.modules import MelSpec
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
    from datasets.arrow_writer import ArrowWriter
    torch.set_num_threads(2)
    vocab, size = get_tokenizer(str(args.vocab),tokenizer='custom')
    if vocab.get(' ') != 0 or size != 2545:
        raise ValueError('unexpected pretrained vocabulary')
    shutil.copyfile(args.vocab,prepared/'vocab.txt')
    mel = MelSpec(mel_spec_type='vocos').eval()
    ids, groups, hashes, texts = set(), {}, {}, {}
    for row in rows:
        identifier = row['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier) or identifier in ids:
            raise ValueError('invalid/duplicate stable id')
        ids.add(identifier)
        split = row['split']
        if split not in ('train','validation','test'):
            raise ValueError('unknown split')
        for mapping,key,label in ((groups,row['group_id'],'group'),(hashes,row['audio_sha256'],'audio'),(texts,canonical_text(row['text']),'text')):
            if key in mapping and mapping[key] != split:
                raise ValueError(f'cross-split {label} leakage: {identifier}')
            mapping[key] = split
        safe_member(row['audio_filepath'])
        if row['audio_filepath'] != f'wavs/{identifier}.wav':
            raise ValueError('id/audio path mismatch')
        if (source/'texts'/f'{identifier}.txt').read_text(encoding='utf-8-sig').strip() != row['text'].strip():
            raise ValueError(f'{identifier}: transcript text file mismatch')
    for split in ('train','validation','test'):
        indexed = records(source/f'{split}.jsonl')
        expected = [{'audio_filepath':r['audio_filepath'],'text':r['text'],'duration':r['duration']} for r in rows if r['split']==split]
        if indexed != expected:
            raise ValueError(f'{split}: source index mismatch')
    wavs = set(p.stem for p in (source/'wavs').glob('*.wav'))
    txts = set(p.stem for p in (source/'texts').glob('*.txt'))
    if wavs != ids or txts != ids:
        raise ValueError('WAV/text file IDs do not exactly match manifest')
    metadata = list(csv.reader((source/'metadata.csv').open(encoding='utf-8-sig',newline=''), delimiter='|'))
    if metadata and metadata[0][0].lower() in ('audio_file','audio_filepath','audio_path','filename','file_name'):
        metadata = metadata[1:]
    if metadata != [[r['id'], r['text'], r['text']] for r in rows]:
        raise ValueError('metadata.csv differs from authoritative IDs and transcripts')
    dump(prepared/'owner-authorization.private.json',{
        'scope':'personal AegisVoice v2 F5 fine-tuning on this SHA256-bound transformed-output corpus',
        'dataset_sha256':receipt['archive_sha256'],'owner_perceptual_approval':True,
        'owner_approval_description':'less expressive, slightly reduced pitch, synthetic/robotic female texture; effects already applied',
        'owner_reports_elevenlabs_email_permission':True,'independently_verified':False,
        'email_archive_status':'private response archiving still required; preserve original reply and headers privately',
        'limitations':'reported permission recorded only for the described personal use; no extrapolation to commercial or other uses',
        'do_not_reapply_effects':True})
    output, audited, quarantine, oov = [], [], [], Counter()
    for i,row in enumerate(rows):
        identifier = row['id']
        path = source/row['audio_filepath']
        actual_hash = digest(path)
        if actual_hash != row['audio_sha256']:
            raise ValueError(f'{identifier}: WAV SHA256 mismatch')
        try:
            values, rate = read_pcm(path)
            info = sf.info(path)
            duration = len(values)/rate
            if info.subtype != 'PCM_24' or info.channels != 1 or rate != 48000:
                raise ValueError('source differs from approved mono48k PCM24 format')
            if abs(duration-row['duration']) > 1/rate or len(values)!=row['frames']:
                raise ValueError('source duration/frames mismatch')
            if not .3 <= duration <= 30 or not np.isfinite(values).all():
                raise ValueError('nonfinite or unsupported duration')
            peak = float(np.abs(values).max())
            rms = float(np.sqrt(np.mean(values**2)))
            clip_count = int(np.sum(np.abs(values)>=1-1/8388608))
            near_clip = int(np.sum(np.abs(values)>=.999))
            if clip_count or peak>=1 or rms<1e-5:
                raise ValueError('clipped/silent source requires review')
            if abs(peak-float(row['peak'])) > 2/8388608 or abs(rms-float(row['rms'])) > 2/8388608:
                raise ValueError('source acoustic provenance mismatch')
            converted = resample_pcm(values,rate,RATE)
            converted_peak = float(np.max(np.abs(converted)))
            if not np.isfinite(converted).all() or converted_peak>=1-1/8388608:
                raise ValueError('resampling overshoot; no clipping applied')
            target = prepared/'audio'/f'{identifier}.wav'
            sf.write(target,converted,RATE,subtype='PCM_24')
            target.chmod(0o600)
            # Cache mel from actual written PCM24 samples, matching inference read path exactly.
            stored,stored_rate=sf.read(target,dtype='float32')
            with torch.inference_mode():
                m=mel(torch.from_numpy(stored).unsqueeze(0)).squeeze(0).numpy()
            if m.ndim!=2 or m.shape[0]!=100 or not np.isfinite(m).all():
                raise ValueError('invalid mel cache')
            mel_path=prepared/'mels'/f'{identifier}.npy'
            np.save(mel_path,m,allow_pickle=False)
            mel_path.chmod(0o600)
            original_text=unicodedata.normalize('NFC',row['text']).strip()
            text=original_text.lower()
            if any(ord(c)<32 for c in text) or not text:
                raise ValueError('invalid transcript')
            tokens=convert_char_to_pinyin([text])[0]
            unknown=Counter(t for t in tokens if t not in vocab)
            oov.update(unknown)
            if unknown:
                raise ValueError(f'OOV tokens: {dict(unknown)}')
            record={'id':identifier,'audio_filepath':str(target),'text':text,'original_text':original_text,
                    'duration':len(stored)/RATE,'split':row['split'],'group_id':row['group_id'],
                    'sha256':digest(target),'source_audio_filepath':str(path),'source_audio_sha256':actual_hash,
                    'mel_filepath':str(mel_path),'mel_shape':list(m.shape),'mel_sha256':digest(mel_path),
                    'tokenized_text':tokens}
            output.append(record)
            audited.append({'id':identifier,'split':row['split'],'source_sha256':actual_hash,'source_rate':rate,
                            'source_frames':len(values),'source_duration':duration,'peak':peak,'rms':rms,
                            'clipped_samples':clip_count,'near_clip_samples':near_clip,'converted_peak':converted_peak,
                            'converted_sha256':record['sha256'],'mel_sha256':record['mel_sha256'],
                            'source_duration_minus_converted':duration-record['duration']})
        except ValueError as exc:
            quarantine.append({'id':identifier,'split':row['split'],'reason':str(exc),'source_audio_sha256':actual_hash})
        if (i+1)%100==0:
            print(json.dumps({'stage':'audit_mels','processed':i+1,'accepted':len(output),'quarantined':len(quarantine)}),flush=True)
    if oov:
        raise ValueError(f'OOV requires review: {dict(oov)}')
    for split in ('train','validation','test'):
        selected=[r for r in output if r['split']==split]
        (prepared/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        d=prepared/'splits'/split
        d.mkdir(mode=0o700,exist_ok=True)
        with ArrowWriter(path=str(d/'raw.arrow')) as writer:
            for r in selected:
                writer.write({'audio_path':r['audio_filepath'],'text':r['tokenized_text'],'duration':r['duration']})
            writer.finalize()
        dump(d/'duration.json',{'duration':[r['duration'] for r in selected]})
        shutil.copyfile(args.vocab,d/'vocab.txt')
    # Conventional top-level raw.arrow contains train alone; held-out splits are separate.
    for name in ('raw.arrow','duration.json'):
        shutil.copyfile(prepared/'splits/train'/name,prepared/name)
    dump(prepared/'audio-audit.private.json',audited)
    dump(prepared/'quarantine.json',quarantine)
    samples=sorted([r for r in output if r['split']=='train'],key=lambda r:r['duration'])
    refs=[]
    for seconds in (6, 9, 12):
        eligible=[r for r in samples if 5 <= r['duration'] <= 12 and r not in refs]
        refs.append(min(eligible, key=lambda r:abs(r['duration']-seconds)))
    dump(prepared/'roundtrip-samples.private.json',refs)
    report={'archive':receipt,'source_manifest_sha256':digest(source/'dataset.jsonl'),
            'source_count':len(rows),'accepted_count':len(output),'quarantine':quarantine,
            'source_duration_seconds':sum(r['duration'] for r in rows),
            'duration_seconds':sum(r['duration'] for r in output),'counts':dict(Counter(r['split'] for r in output)),
            'split_seconds':{s:sum(r['duration'] for r in output if r['split']==s) for s in ('train','validation','test')},
            'all_source_wav_hashes_verified':True,'source_wavs':dict(Counter(f'{r["sample_rate"]}Hz/{r["subtype"]}/{r["channels"]}ch' for r in rows)),
            'source_clipped_samples':sum(r['clipped_samples'] for r in audited),
            'source_near_clip_samples':sum(r['near_clip_samples'] for r in audited),
            'group_cross_split_leakage':0,'exact_audio_cross_split_leakage':0,'normalized_text_cross_split_leakage':0,
            'split_indexes_match_manifest':True,'id_wav_text_sets_match':True,
            'metadata_csv_rows':len(metadata),'metadata_csv_matches_manifest':True,
            'transcript_alignment_scope':'exact dataset/index/text/CSV consistency; not word-level listening or independent ASR of all clips',
            'vocab_size':size,'vocab_sha256':digest(args.vocab),'oov_tokens':dict(oov),
            'text_transform':'NFC/strip and explicit lowercase for F5, original_text preserved; official convert_char_to_pinyin frontend',
            'audio_transform':'48k PCM24 mono -> scipy resample_poly Kaiser beta10 ->24k PCM24 mono; no gain/trim/denoise/pitch/speed/effects',
            'mel_cache':'exact pinned F5 MelSpec(vocos) CPU float32 from written24k PCM24, [100,T], npy allow_pickle=False',
            'training_arrow':'top-level raw.arrow=training only; splits/{train,validation,test} each isolated arrow+durations+vocab',
            'elapsed_seconds':time.monotonic()-started,'test_policy':'fixed source test records excluded from training and convergence decisions',
            'versions':{'numpy':np.__version__,'torch':torch.__version__,'soundfile':sf.__version__},
            'manifest_hashes':{s:digest(prepared/f'{s}.jsonl') for s in ('train','validation','test')}}
    dump(prepared/'audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--f5-source',type=Path,required=True)
    p.add_argument('--vocab',type=Path,required=True)
    prepare(p.parse_args())

if __name__=='__main__':
    main()
