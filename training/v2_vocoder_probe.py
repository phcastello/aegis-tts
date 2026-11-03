"""Pinned BigVGAN reconstruction on held-out Aegis clips, no normalization."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/var/tmp/aegis-v2-work'));p.add_argument('--count',type=int,default=3);a=p.parse_args()
 os.umask(0o077);root=a.root;sys.path.insert(0,str(root/'bigvgan/source'))
 import numpy as np,soundfile as sf,torch
 from bigvgan import BigVGAN,load_hparams_from_json
 from meldataset import get_mel_spectrogram
 torch.set_num_threads(2)
 h=load_hparams_from_json(root/'bigvgan/config.json');v=BigVGAN(h,use_cuda_kernel=False)
 v.load_state_dict(torch.load(root/'bigvgan/bigvgan_generator.pt',map_location='cpu',weights_only=True)['generator'],strict=True);v.remove_weight_norm();v=v.cuda().eval()
 rows=[json.loads(x) for x in Path('/var/tmp/aegisvoice-v2-training-20261102/prepared/validation.jsonl').read_text().splitlines()][:a.count]
 out=root/'bigvgan/roundtrip';out.mkdir(exist_ok=True);results=[]
 for row in rows:
  y,sr=sf.read(row['audio_filepath'],dtype='float32');assert sr==h.sampling_rate
  y=torch.from_numpy(y)[None].cuda();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  with torch.inference_mode():
   mel=get_mel_spectrogram(y,h);z=v(mel).squeeze().cpu().numpy()
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  path=out/(row['id']+'.wav');sf.write(path,z,sr,subtype='FLOAT')
  m2=get_mel_spectrogram(torch.from_numpy(z)[None].cuda(),h);n=min(mel.shape[-1],m2.shape[-1])
  r=dict(id=row['id'],path=str(path),source=row['audio_filepath'],text=row['text'],seconds=len(z)/sr,total_s=elapsed,rtf=elapsed/(len(z)/sr),peak=float(abs(z).max()),mel_mae=float((mel[...,:n]-m2[...,:n]).abs().mean()),allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  results.append(r);print(json.dumps(r),flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
