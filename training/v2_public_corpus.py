"""Prepare authors' TTS-Portuguese ZIP without expanded originals or acoustic cleanup.

Public CC BY4.0 corpus: https://github.com/Edresson/TTS-Portuguese-Corpus .
Full source ZIP retained; SHA256/CRC verified while decoding selected waveforms.
Adjacent batches of 50 utterances and exact text/audio duplicates remain together.
"""
import argparse,concurrent.futures,hashlib,io,json,os,re,unicodedata,zipfile
from pathlib import Path

def norm(x):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFC',x).casefold()))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def process(args):
 archive,name,text,out,max_seconds=args
 import numpy as np,soundfile as sf
 from scipy.signal import resample_poly
 with zipfile.ZipFile(archive) as z:raw=z.read('TTS-Portuguese-Corpus/'+name)
 a,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
 ident=Path(name).stem;seconds=len(a)/sr;peak=float(abs(a).max());r=dict(id=ident,original_text=text,text=unicodedata.normalize('NFC',text).strip().lower(),source_archive_member=name,source_audio_sha256=hashlib.sha256(raw).hexdigest(),source_rate=sr,duration=seconds,peak=peak,source_channels=a.shape[1],group_id='source-batch-'+str(int(ident.split('-')[-1])//50))
 if not 1<=seconds<=max_seconds or a.shape[1]!=1 or not np.isfinite(a).all() or not text.strip() or peak>=1:
  return dict(**r,rejected='duration/channel/finite/text/clip gate')
 assert sr==48000, sr
 y=resample_poly(a[:,0],1,2,window=('kaiser',10.0));dest=Path(out)/'audio'/(ident+'.wav');sf.write(dest,y,24000,subtype='PCM_24')
 r.update(audio_filepath=str(dest),sha256=sha(dest),duration=len(y)/24000,prepared_rate=24000,source_seconds=seconds)
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/var/tmp/aegis-v2-work/portuguese'));p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-seconds',type=float,default=22);a=p.parse_args();os.umask(0o077)
 archive=a.root/'corpus.zip';partial=a.root/'corpus.zip.part'
 if not archive.exists():
  with zipfile.ZipFile(partial) as z:assert z.infolist()
  partial.rename(archive)
 out=a.output or a.root/'prepared'
 if (out/'audit.json').exists():raise ValueError('Prepared corpus already audited; use a new --output instead of overwriting manifests')
 (out/'audio').mkdir(parents=True,exist_ok=True,mode=0o700)
 with zipfile.ZipFile(archive) as z:
  names=set(z.namelist());text=z.read('TTS-Portuguese-Corpus/texts.csv').decode('utf-8-sig');pairs=[x.split('==',1) for x in text.splitlines() if x.strip()]
  # The published CSV contains repeated IDs, including conflicting texts. Never
  # run workers against the same output path or silently select one transcript.
  grouped={}
  for n,t in pairs:grouped.setdefault(n,[]).append(t)
  ambiguous={n:ts for n,ts in grouped.items() if len({norm(t) for t in ts})>1}
  tasks=[(str(archive),n,ts[0],str(out),a.max_seconds) for n,ts in grouped.items()
         if n not in ambiguous and 'TTS-Portuguese-Corpus/'+n in names]
  missing=[n for n,t in pairs if 'TTS-Portuguese-Corpus/'+n not in names]
  (out/'ambiguous-source-transcripts.json').write_text(json.dumps(ambiguous,ensure_ascii=False,indent=2))
 records=[]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for i,r in enumerate(pool.map(process,tasks)):
   records.append(r)
   if i%100==0:print(json.dumps(dict(event='public_prepared',count=i,total=len(tasks))),flush=True)
 # Union adjacent source batches when exact audio/text duplicate links them.
 parent={r['group_id']:r['group_id'] for r in records};seen={}
 def find(k):
  while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
  return k
 for r in records:
  for key in ('text:'+norm(r['text']),'audio:'+r['source_audio_sha256']):
   if key in seen:parent[find(r['group_id'])]=find(seen[key])
   seen[key]=r['group_id']
 parts={'train':[],'validation':[],'test':[]};reject=[]
 for r in records:
  if 'rejected' in r:reject.append(r);continue
  r['group_id']=find(r['group_id']);bucket=int(hashlib.sha256(('20261103/'+r['group_id']).encode()).hexdigest()[:8],16)%100
  part='validation' if bucket<5 else 'test' if bucket<10 else 'train';parts[part].append(r)
 for part,rows in parts.items():(out/(part+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 report=dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,metadata_lines=len(pairs),present_unambiguous_audio=len(tasks),missing_audio=missing,ambiguous_source_ids=list(ambiguous),identical_metadata_duplicates_deduplicated=sum(len(ts)-1 for n,ts in grouped.items() if n not in ambiguous),selected_max_seconds=a.max_seconds,selection='full utterances; quarantine conflicting source IDs; deduplicate identical metadata; exclude <1s/>22s, clipped/multichannel/nonfinite/empty; no denoise/trim/gain/pitch/speed',resampling='48k->24k resample_poly Kaiser10, PCM24',split='sha256 seed20261103 over union components of adjacent50 ids + duplicate text/audio',rejected=len(reject),rejected_seconds=sum(r['duration'] for r in reject),splits={k:dict(count=len(v),seconds=sum(r['duration'] for r in v)) for k,v in parts.items()})
 (out/'rejections.json').write_text(json.dumps(reject,ensure_ascii=False));(out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
