"""Proposed fake-only acoustic reference contracts; no Torch or model load."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gateway'))
_spec=importlib.util.spec_from_file_location('aegis_acoustic_reference_runtime_tests',ROOT/'gateway/voxcpm_runtime.py')
runtime=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=runtime;_spec.loader.exec_module(runtime)


def config(tmp_path,**kwargs):
    return runtime.VoxConfig(source=tmp_path/'source',model=tmp_path/'model',adapter=tmp_path/'adapter.pt',
        adapter_sha256='0'*64,cache=tmp_path/'cache',gpu_lock=tmp_path/'gpu.lock',**kwargs)


class Feature:
    shape=(64,4,64);dtype='fp32';device=SimpleNamespace(type='cpu');finite=True


def session(tmp_path,*,reference=True,tokens=3):
    path=tmp_path/'reference.wav';path.write_bytes(b'synthetic bytes; never decoded')
    cfg=config(tmp_path,**({'reference_wav':path,'reference_sha256':hashlib.sha256(path.read_bytes()).hexdigest()} if reference else {}))
    value=runtime.NativeSession(cfg);calls=[];feature=Feature()
    cache={'mode':'reference','ref_audio_feat':feature}
    value.torch=SimpleNamespace(float32='fp32',is_tensor=lambda obj:isinstance(obj,Feature),
        isfinite=lambda obj:SimpleNamespace(all=lambda:obj.finite),
        cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,empty_cache=lambda:None),
        compiler=SimpleNamespace(cudagraph_mark_step_begin=lambda:None))
    value.model=SimpleNamespace(text_tokenizer=lambda text:list(range(tokens)))
    value.check_residency=lambda:None;value.check_budget=lambda:None;value.observe=lambda:{}
    def build(**kwargs):calls.append(kwargs);return cache
    value.model.build_prompt_cache=build
    return value,cache,calls


def test_optional_config_pair_and_read_keep_none_profile_unchanged(tmp_path):
    cfg=config(tmp_path)
    assert cfg.reference_wav is None and cfg.reference_sha256 is None
    assert cfg.profile=='AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-OriginalNFC-Experimental'
    for kwargs in ({'reference_wav':tmp_path/'ref.wav'},{'reference_sha256':'0'*64},
                   {'reference_wav':Path('relative.wav'),'reference_sha256':'0'*64},
                   {'reference_wav':tmp_path/'ref.wav','reference_sha256':'bad'}):
        with pytest.raises(ValueError):config(tmp_path,**kwargs)
    raw={key:str(getattr(cfg,key)) for key in ('source','model','adapter','cache','gpu_lock')}
    raw['adapter_sha256']='0'*64
    path=tmp_path/'config.json';path.write_text(json.dumps(raw))
    assert runtime.VoxConfig.read(path).reference_wav is None
    raw.update(reference_wav=str(tmp_path/'ref.wav'),reference_sha256='1'*64);path.write_text(json.dumps(raw))
    assert runtime.VoxConfig.read(path).reference_wav==tmp_path/'ref.wav'



def test_reference_profile_distinguishes_conditioning_without_private_details(tmp_path):
    off=config(tmp_path)
    on=config(tmp_path,reference_wav=tmp_path/'private-reference.wav',reference_sha256='a'*64)
    assert on.profile!=off.profile
    assert on.profile=='AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-OriginalNFC-Experimental-AcousticReference'
    assert on.profile==off.profile+'-AcousticReference'
    assert str(on.reference_wav) not in on.profile and on.reference_sha256 not in on.profile


def test_exact_no_reference_and_reference_capacity_boundaries(tmp_path):
    plain=config(tmp_path)
    assert runtime.capacity_guard(622,plain)==dict(text_tokens=622,reserved_slots=1024,capacity=1024,audio_patch_limit=400)
    with pytest.raises(runtime.BackendTextTooLong):runtime.capacity_guard(623,plain)
    ref=config(tmp_path,reference_wav=tmp_path/'ref.wav',reference_sha256='0'*64)
    assert runtime.capacity_guard(556,ref)['reserved_slots']==1024
    assert runtime.capacity_guard(556,ref)['reference_patches']==64
    with pytest.raises(runtime.BackendTextTooLong):runtime.capacity_guard(557,ref)


def test_none_preparation_does_not_touch_torch_reference_or_files(tmp_path,monkeypatch):
    value=runtime.NativeSession(config(tmp_path))
    monkeypatch.setattr(runtime,'digest',lambda path:pytest.fail('None must not hash a reference'))
    value._prepare_reference()
    assert value.prompt_cache is None and value.reference_cache_build_seconds==0.


def test_reference_build_is_once_and_never_silence_trimmed(tmp_path):
    value,cache,calls=session(tmp_path)
    value._prepare_reference();value._prepare_reference()
    assert value.prompt_cache is cache and len(calls)==1
    assert calls[0]==dict(reference_wav_path=str(value.config.reference_wav),trim_silence_vad=False)
    assert value.reference_cache_build_seconds>=0


@pytest.mark.parametrize('change',['shape','dtype','device','finite','mode','caption'])
def test_invalid_reference_contract_fails_before_ready(tmp_path,change):
    value,cache,_=session(tmp_path)
    if change=='shape':cache['ref_audio_feat'].shape=(63,4,64)
    elif change=='dtype':cache['ref_audio_feat'].dtype='bf16'
    elif change=='device':cache['ref_audio_feat'].device=SimpleNamespace(type='cuda')
    elif change=='finite':cache['ref_audio_feat'].finite=False
    elif change=='mode':cache['mode']='continuation'
    else:cache['prompt_text']='caption forbidden'
    with pytest.raises(ValueError):value._prepare_reference()
    assert value.prompt_cache is None


def test_reference_change_during_encoding_is_rejected(tmp_path):
    value,cache,_=session(tmp_path)
    def changed(**kwargs):value.config.reference_wav.write_bytes(b'changed');return cache
    value.model.build_prompt_cache=changed
    with pytest.raises(ValueError,match='changed during'):value._prepare_reference()
    assert value.prompt_cache is None


class Iterator:
    def __init__(self,items):self.items=iter(items);self.closed=0
    def __iter__(self):return self
    def __next__(self):
        value=next(self.items)
        if isinstance(value,Exception):raise value
        return value
    def close(self):self.closed+=1


def test_none_uses_legacy_audio_api_and_same_parameters(tmp_path):
    value,_,_=session(tmp_path,reference=False);calls=[]
    value.model.generate_streaming=lambda **kwargs:calls.append(kwargs) or iter(['legacy-audio'])
    assert list(value.generate('Texto.'))==['legacy-audio']
    assert calls==[dict(target_text='Texto.',max_len=400,inference_timesteps=10,cfg_value=2.,retry_badcase=False,
        retry_badcase_ratio_threshold=6.,trim_silence_vad=False,streaming_prefix_len=4,seed=40733133)]


def test_reference_tuple_audio_only_reuses_cache_but_prefills_each_request(tmp_path):
    value,cache,_=session(tmp_path);value.prompt_cache=cache;calls=[];streams=[]
    def generate(**kwargs):
        calls.append(kwargs);stream=Iterator([('audio','tokens',['features'])]);streams.append(stream);return stream
    value.model.generate_with_prompt_cache_streaming=generate
    text='Aegis, leia às 14h05.'
    for _ in range(2):assert list(value.generate(text))==['audio']
    assert len(calls)==2 and all(row['prompt_cache'] is cache for row in calls)
    assert all(row==dict(target_text=text,prompt_cache=cache,max_len=400,inference_timesteps=10,cfg_value=2.,
        retry_badcase=False,retry_badcase_ratio_threshold=6.,streaming_prefix_len=4,seed=40733133) for row in calls)
    assert all(stream.closed==1 for stream in streams)


@pytest.mark.parametrize('items',[[('first',None,None),('second',None,None)],
                                 [('first',None,None),RuntimeError('native error')],
                                 [('first',None,None),('bad','tuple')]])
def test_reference_close_propagates_to_actual_native_iterator_on_cancel_or_error(tmp_path,items):
    value,cache,_=session(tmp_path);value.prompt_cache=cache;native=Iterator(items)
    value.model.generate_with_prompt_cache_streaming=lambda **kwargs:native
    stream=value.generate('Texto.');assert next(stream)=='first'
    if isinstance(items[1],RuntimeError):
        with pytest.raises(RuntimeError):next(stream)
    elif len(items[1])==2:
        with pytest.raises(ValueError):next(stream)
    else:stream.close()
    assert native.closed==1


def test_reference_overflow_and_missing_cache_do_not_enter_native(tmp_path):
    value,cache,_=session(tmp_path,tokens=557)
    value.model.generate_with_prompt_cache_streaming=lambda **kwargs:pytest.fail('overflow entered native')
    with pytest.raises(RuntimeError,match='not ready'):value.generate('Texto.')
    value.prompt_cache=cache
    with pytest.raises(runtime.BackendTextTooLong):value.generate('Texto.')
