"""Freeze FastPitch Aegis metadata only; never read audio/tensors or the final split."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import unicodedata

ROOT=Path('/var/tmp/aegis-v2-work')
RECIPE_SHA='453c3806b384c0af8aa0fdc1cc79b97282b6940f971cc84ac01ed35c1011705d'
FROZEN_SHA={'train':'29c81ccc3cf2df9c83fef5e58f8bdf6adf830e5de5f746942846da0b5ce79b6e',
            'validation':'d330232876238d8a5c4c9065384a8856fb9b4cbdc4bc7a44d729fb2d2fde864b'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def normalize(text):
    text=unicodedata.normalize('NFC',text).lower()
    text=text.translate(str.maketrans({'“':"'",'”':"'",'‘':"'",'’':"'",'\"':"'",'–':'-','—':'-','−':'-','·':'.','…':'...'}))
    return ' '.join(text.split())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('/var/tmp/aegisvoice-v2-training-20261102/prepared'))
    p.add_argument('--legacy-features',type=Path,default=ROOT/'fastpitch/target-features')
    p.add_argument('--output',type=Path,default=ROOT/'fastpitch/aegis-preparation')
    a=p.parse_args();os.umask(0o077)
    if any((parent/'.git').exists() for parent in (a.output,*a.output.parents)):p.error('Private manifests must stay outside Git')
    recipe=Path(__file__).with_name('fastpitch_experiment.py')
    if sha(recipe)!=RECIPE_SHA:raise ValueError('Reviewed recipe changed')
    feature_config=json.loads((a.legacy_features/'features.json').read_text())
    public_config=json.loads((ROOT/'fastpitch/portuguese-features-clean-v1/features.json').read_text())
    for key in ('vocab','mel','upstream','pitch_method'):
        if feature_config[key]!=public_config[key]:raise ValueError('Legacy/public acoustic contract differs: '+key)
    frozen={};canonical={};seen={kind:{} for kind in ('id','group','audio','text')};summary={}
    for split,expected_sha in FROZEN_SHA.items():
        source=a.prepared/(split+'.jsonl')
        if sha(source)!=expected_sha:raise ValueError('Frozen manifest changed: '+split)
        frozen[split]=rows(source);canonical[split]=[];oov=collections.Counter()
        for row in frozen[split]:
            text=unicodedata.normalize('NFC',row['original_text']);effective=normalize(text)
            for kind,key in (('id',row['id']),('group',row['group_id']),('audio',row['source_audio_sha256']),('text',effective)):
                if key in seen[kind] and seen[kind][key]!=split:raise ValueError('Cross-split overlap: '+kind)
                if kind=='id' and key in seen[kind]:raise ValueError('Duplicate ID')
                seen[kind][key]=split
            audio=Path(row['source_audio_filepath']);audio_stat=audio.lstat()
            if audio.is_symlink() or not stat.S_ISREG(audio_stat.st_mode):raise ValueError('Original audio is not a real file')
            oov.update(ch for ch in effective if ch not in feature_config['vocab'])
            canonical[split].append(dict(id=row['id'],split=split,group_id=row['group_id'],text=text,
                original_text=row['original_text'],audio_filepath=str(audio),source_audio_sha256=row['source_audio_sha256'],
                duration=row['duration'],frozen_manifest_sha256=expected_sha))
        if oov:raise ValueError('OOV characters: '+repr(oov))
        summary[split]=dict(count=len(frozen[split]),seconds=sum(r['duration'] for r in frozen[split]),oov={},
            tokens=sum(len(normalize(r['original_text']))+2 for r in frozen[split]),
            max_tokens=max(len(normalize(r['original_text']))+2 for r in frozen[split]),
            longest=max((dict(id=r['id'],seconds=r['duration']) for r in frozen[split]),key=lambda r:r['seconds']))
    if {s:len(r) for s,r in frozen.items()}!={'train':1025,'validation':55}:raise ValueError('Unexpected frozen selection size')
    wanted={r['id']:r for selected in canonical.values() for r in selected};legacy_source={};legacy_rows=[]
    for path,expected_sha in feature_config['source_manifests'].items():
        if sha(path)!=expected_sha:raise ValueError('Legacy source manifest changed')
        for row in rows(path):
            if row['id'] in wanted:legacy_source[row['id']]=row
    for split in ('train','validation'):legacy_rows.extend(rows(a.legacy_features/(split+'.jsonl')))
    reusable=[];mismatch=[];all_bytes=0;all_blocks=0;all_frames=0
    for row in legacy_rows:
        path=Path(row['feature']);st=path.lstat()
        if path.is_symlink() or not stat.S_ISREG(st.st_mode):raise ValueError('Legacy feature is not a real file')
        all_bytes+=st.st_size;all_blocks+=st.st_blocks*512;all_frames+=row['frames']
        if row['id'] not in wanted:continue
        new=wanted[row['id']];old=legacy_source[row['id']]
        matches=(Path(old['audio_filepath']).resolve()==Path(new['audio_filepath']).resolve()
                 and old.get('source_sha256')==new['source_audio_sha256'] and old['group_id']==new['group_id']
                 and normalize(old['text'])==normalize(new['text']) and row['text_length']==len(normalize(new['text']))+2)
        if not matches:mismatch.append(row['id']);continue
        reusable.append(dict(id=row['id'],split=new['split'],source=str(path),feature_sha256=row['feature_sha256'],
            frames=row['frames'],logical_bytes=st.st_size,allocated_bytes=st.st_blocks*512))
    if mismatch:raise ValueError('Selected legacy metadata mismatch: '+repr(mismatch))
    cache_ids={r['id'] for r in reusable};missing=[r['id'] for selected in canonical.values() for r in selected if r['id'] not in cache_ids]
    estimated_frames=sum(math.floor(r['duration']*24000/256) for r in wanted.values())
    estimated_bytes=math.ceil(all_bytes*estimated_frames/all_frames)+1080*4096
    historical=json.loads((ROOT/'fastpitch/target-prepare-status.json').read_text())
    historical_wall=(datetime.datetime.fromisoformat(historical['ended_utc'])-datetime.datetime.fromisoformat(historical['started_utc'])).total_seconds()
    a.output.mkdir(parents=True,exist_ok=True)
    for split,selected in canonical.items():
        target=a.output/(split+'.jsonl');temporary=target.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream:
            for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
    write(a.output/'reuse-plan.json',dict(status='metadata_only_not_linked_or_tensor_verified',reusable=reusable,missing_ids=missing))
    plan=dict(status='prepared_metadata_only',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recipe_sha256=RECIPE_SHA,
        helper_sha256=sha(__file__),frozen_manifest_sha256=FROZEN_SHA,manifests={s:dict(path=str(a.output/(s+'.jsonl')),sha256=sha(a.output/(s+'.jsonl'))) for s in canonical},
        splits=summary,acoustic_contract={k:feature_config[k] for k in ('vocab','mel','upstream','pitch_method')},
        input_policy='Original48kPCM24 mono by path/SHA; unchanged NFC original_text; recipe resamples in RAM, no gain/trim/DSP; final split never opened',
        legacy_cache=dict(count=len(legacy_rows),logical_bytes=all_bytes,allocated_bytes=all_blocks,reusable_count=len(reusable),missing_count=len(missing),
            excluded_from_new_selection=len(legacy_rows)-len(reusable),reuse_status='Needs whole feature SHA and audio/tensor identity audit before acceptance',
            old_pitch_mean_hz=feature_config['pitch_mean'],old_pitch_std_hz=feature_config['pitch_std'],old_split='810train/180validation; never reuse old split/statistics'),
        storage=dict(estimated_full_feature_bytes=estimated_bytes,estimated_increment_after_verified_hardlinks_bytes=estimated_bytes-sum(r['logical_bytes'] for r in reusable),
            current_volume_free_bytes=shutil.disk_usage(a.output).free,feature_temp_margin_bytes=32*1024**2,checkpoint_bytes_reference=549622954,
            future_new_training_last_best_atomic_overlap_bytes=3*549622954,global_budget='Refresh physical global usage before authorized work; <=40GB and >=5GiB free after peak'),
        timing=dict(historical_legacy_4workers_wall_seconds=historical_wall,full_2workers_estimate_minutes=[5,8],verified_reuse_2workers_estimate_minutes=[1,3],
            estimate_scope='CPU extrapolation only; no new extraction or benchmark executed; account for concurrent training/ASR'),
        pitch_policy='Store WORLD Hz labels; normalize voiced only with new1025train mean/std; keep unvoiced0. Recompute statistics and refill model buffers after strict init. Infer outputs normalized coordinates unless explicit pitch_transform; never treat as raw Hz.',
        initialization='Future qualified public checkpoint only; --init-checkpoint, no --resume: strict model, fresh Adam/scheduler/scaler/RNG/data position. New target output directory; no language/identity approval from CTC or mel.',
        next_action='Root review, budget refresh, then selected hardlink SHA check + prepare missing/recompute train-only stats + full CPU audit; no work started')
    write(a.output/'plan.json',plan)
    print(json.dumps({k:plan[k] for k in ('status','splits','legacy_cache','storage','timing')},ensure_ascii=False))


if __name__=='__main__':main()
