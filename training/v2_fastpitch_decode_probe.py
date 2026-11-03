"""CPU-only decoding of saved text-only FastPitch validation mels.

This diagnostic never occupies the GPU or reports product latency. Teacher-
forced predictions are intentionally excluded. Full GPU inference is measured
separately after loading the selected acoustic and vocoder weights together.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,default=Path('/var/tmp/aegis-v2-work/portuguese/fastpitch-input-v1'));p.add_argument('--vocoder',type=Path,default=Path('/var/tmp/aegis-v2-work/bigvgan'));a=p.parse_args();os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']=''
 a.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 import torch,numpy as np,soundfile as sf
 torch.set_num_threads(2);source=a.run/'validation-mels.pt';frozen=a.output/'native-mels.pt'
 # Opening the atomic source retains a single complete inode across replacement.
 with source.open('rb') as f:raw=f.read()
 frozen.write_bytes(raw);del raw;packet=torch.load(frozen,map_location='cpu',weights_only=True)
 rows={r['id']:r for r in map(json.loads,(a.inputs/'validation.jsonl').read_text(encoding='utf-8').splitlines())}
 sys.path.insert(0,str(a.vocoder/'source'));from bigvgan import BigVGAN,load_hparams_from_json
 v=BigVGAN(load_hparams_from_json(a.vocoder/'config.json'),use_cuda_kernel=False)
 v.load_state_dict(torch.load(a.vocoder/'bigvgan_generator.pt',map_location='cpu',weights_only=True)['generator'],strict=True);v.remove_weight_norm();v.eval();result=[];jobs=[]
 for sample in packet['examples']:
  if 'native_mel' not in sample:continue
  mel=sample['native_mel'];assert mel.ndim==3 and mel.shape[1]==100 and torch.isfinite(mel).all()
  start=time.monotonic()
  with torch.inference_mode():audio=v(mel).squeeze().numpy()
  assert audio.size and np.isfinite(audio).all();path=a.output/(sample['id']+'.wav');sf.write(path,audio,24000,subtype='FLOAT')
  row=dict(id=sample['id'],step=packet['step'],native_frames=sample['native_frames'],path=str(path),sha256=sha(path),seconds=len(audio)/24000,peak=float(abs(audio).max()),samples_at_generator_bound=int((abs(audio)>=1).sum()),cpu_decode_wall_seconds=time.monotonic()-start)
  result.append(row);write(a.output/'results.json',dict(step=packet['step'],scope='CPU diagnostic from text-only native mels; not inference/TTFA benchmark',mel_snapshot_sha256=sha(frozen),vocoder='BigVGANv2 24k100band256x; official output clamp [-1,1] recorded by bound count',results=result))
  jobs.append(dict(candidate='fastpitch-public-step'+str(packet['step']),case=sample['id'],wav=str(path),expected_verbatim=rows[sample['id']]['text'],cohort='public-four-native-validation'))
  write(a.output/'asr-jobs.json',dict(jobs=jobs));print(json.dumps(row),flush=True)
 write(a.output/'completed.json',dict(step=packet['step'],count=len(result),cuda_initialized=torch.cuda.is_initialized()))
if __name__=='__main__':main()
