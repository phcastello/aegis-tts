"""Read-only model/cache inventory, SHA256 and safetensors header inspection."""
from __future__ import annotations
from collections import Counter
import argparse,datetime,hashlib,json,os,struct
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def architecture(value):
    fields={'model_type','architectures','torch_dtype','dtype','hidden_size','intermediate_size','num_hidden_layers',
            'num_attention_heads','num_key_value_heads','vocab_size','max_position_embeddings','quantization_config',
            'transformers_version','tokenizer_class','sample_rate','sampling_rate','input_sample_rate','output_sample_rate',
            'num_code_groups','num_quantizers','num_codebooks','codec_type','quantizer_type','_commit_hash'}
    out={k:v for k,v in value.items() if k in fields}
    for k,v in value.items():
        if isinstance(v,dict) and ('config' in k or k in ('encoder','decoder')):
            selected=architecture(v)
            if selected:out[k]=selected
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('/home/phc/aegis-tts'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    root=a.repo/'models/Qwen3-TTS-12Hz-0.6B-Base';files=[];inodes={};errors=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        stat=path.stat();key=(stat.st_dev,stat.st_ino)
        with path.open('rb') as stream:first=stream.read(256)
        pointer=first.startswith(b'version https://git-lfs.github.com/spec/v1')
        item=dict(path=str(path.relative_to(root)),realpath=str(path.resolve()),bytes=stat.st_size,allocated_bytes=stat.st_blocks*512,
                  device=stat.st_dev,inode=stat.st_ino,owner_uid=stat.st_uid,is_symlink=path.is_symlink(),lfs_pointer=pointer)
        if pointer:errors.append('LFS pointer: '+item['path'])
        if key not in inodes:inodes[key]=sha(path)
        item['sha256']=inodes[key]
        if path.suffix=='.json' and 'config' in path.name:
            try:item['architectural_config']=architecture(json.loads(path.read_text()))
            except Exception as e:item['parse_error']=repr(e)
        if path.suffix=='.safetensors' and not pointer:
            with path.open('rb') as stream:
                length=struct.unpack('<Q',stream.read(8))[0]
                if length>16*1024**2:raise ValueError('unexpected safetensors header')
                header=json.loads(stream.read(length))
            dtypes=Counter();params=Counter();elements=0
            for name,tensor in header.items():
                if name=='__metadata__':continue
                count=1
                for dim in tensor['shape']:count*=dim
                dtypes[tensor['dtype']]+=1;params[tensor['dtype']]+=count;elements+=count
                if tensor['data_offsets'][1]+length+8>stat.st_size:errors.append('tensor beyond EOF: '+item['path']+'/'+name)
            item['safetensors']=dict(header_bytes=length,tensors=sum(dtypes.values()),dtypes=dict(dtypes),elements_by_dtype=dict(params),elements=elements,
                                     quantization_tensor_names=[n for n in header if any(w in n.lower() for w in ('qweight','qzeros','weight_scale','scales'))][:30])
        if path.suffix=='.metadata' and '/huggingface/' in str(path):
            item['hf_download_metadata_lines']=path.read_text().splitlines()[:3]
        files.append(item)
    caches=[]
    for directory in sorted((a.repo/'cache').iterdir()):
        if not directory.is_dir():continue
        seen=set();allocated=logical=0;count=0;refs=[]
        for path in directory.rglob('*'):
            try:
                if not path.is_file():continue
                stat=path.stat();key=(stat.st_dev,stat.st_ino);count+=1
                if key not in seen:allocated+=stat.st_blocks*512;logical+=stat.st_size;seen.add(key)
                if '/refs/' in str(path) and stat.st_size<=200:
                    content=path.read_text().strip()
                    if len(content) in (40,64) and all(c in '0123456789abcdef' for c in content):refs.append(dict(path=str(path),revision=content))
            except (PermissionError,OSError):continue
        caches.append(dict(path=str(directory),files=count,unique_files=len(seen),allocated_bytes=allocated,logical_bytes=logical,hf_refs=refs,
                           scope='cache names/sizes/refs only; no token, credential or private prompt contents read'))
    report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_root=str(root),model_id='Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                files=files,errors=errors,model_logical_bytes=sum(i['bytes'] for i in files),
                unique_model_files=len(inodes),caches=caches,
                completeness='All model files fully hashed and header-checked, not loaded. No GPU, service or protected reference changes.',
                quantization='Inspect safetensors dtypes and actual config; BF16 storage alone is not an integer quantization claim.',
                related_model_preserved=str(a.repo/'models/Qwen3-TTS-12Hz-1.7B-Base'))
    a.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True);temporary=a.output.with_suffix('.new')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temporary.replace(a.output)
    print(json.dumps(dict(output=str(a.output),files=len(files),bytes=report['model_logical_bytes'],errors=errors,
                         weights=[{k:i[k] for k in ('path','bytes','sha256','safetensors')} for i in files if 'safetensors' in i])))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
