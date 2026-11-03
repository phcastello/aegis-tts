"""CPU/fake-model contracts for the optional gateway. No Torch/CUDA/model load."""
from __future__ import annotations
import asyncio
from contextlib import aclosing
import io
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave

import httpx
import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gateway'))
import default_reference_gateway as gateway
# Historical file probes intentionally used the same absolute module name.
# Load this test's gateway runtime by exact path without replacing that module
# for legacy tests; only the Vox lifespan fixture temporarily installs it.
_spec=importlib.util.spec_from_file_location('aegis_gateway_voxcpm_runtime_tests',
    Path(__file__).resolve().parents[1]/'gateway/voxcpm_runtime.py')
runtime=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=runtime
_spec.loader.exec_module(runtime)
from voxcpm_adapter import processed_stream
from voxcpm_audio import HalfRateFIR,pcm16_bytes
from aegis_speech.experimental_text import OriginalTextNormalizer,OriginalTextSegmenter,RefinedTextNormalizer
from output_onset import OutputOnsetTrimmer


def config(tmp_path,**kwargs):
    return runtime.VoxConfig(source=tmp_path/'source',model=tmp_path/'model',adapter=tmp_path/'adapter.pt',
        adapter_sha256='0'*64,cache=tmp_path/'cache',gpu_lock=tmp_path/'gpu.lock',**kwargs)


class Session:
    def __init__(self,cfg,records,gate=None):
        self.cfg=cfg;self.records=records;self.gate=gate;self.steps=[];self.closed_streams=0;self.inputs=[]
        t=np.arange(10003)/48000
        self.audio=(.2*np.sin(2*np.pi*173*t)).astype(np.float32)

    def record(self,event):self.records.append((event,threading.get_ident()))
    def load(self):
        self.record('load')
        if self.gate is not None and not self.gate.wait(3):raise TimeoutError('fixture release missing')
        return self
    def token_count(self,text):return 623 if text.startswith('overflow') else max(1,len(text))
    def reset(self):self.record('reset')
    def close(self):self.record('close')
    def generate(self,text):
        self.record('generate')
        self.inputs.append(text)
        try:
            if text=='many':
                for index in range(10000):
                    self.steps.append(index)
                    if index==1 and getattr(self,'generation_gate',None) is not None:self.generation_gate.wait(3)
                    yield np.full(480,.1,dtype=np.float32)
            elif text=='empty':
                yield np.empty(0,dtype=np.float32)
            elif text=='invalid':
                yield np.array([np.nan],dtype=np.float32)
            elif text=='silêncio':
                def quiet(ms):return np.zeros(ms*48,dtype=np.float32)
                def tone(ms):return (.2*np.sin(2*np.pi*173*np.arange(ms*48)/48000)).astype(np.float32)
                y=np.r_[quiet(430),tone(120),quiet(650),tone(90),quiet(180)]
                for start in range(0,len(y),4800):yield y[start:start+4800]
            else:
                offset=0
                for size in (7,41,3001,6954):
                    yield self.audio[offset:offset+size];offset+=size
        finally:self.closed_streams+=1;self.record('generator_close')


async def started(tmp_path,gate=None,**kwargs):
    records=[];holder=[]
    def factory(cfg):
        value=Session(cfg,records,gate);holder.append(value);return value
    backend=runtime.VoxBackend(config(tmp_path,**kwargs),factory)
    await backend.start()
    if gate is None:await asyncio.wait_for(backend.ready_event.wait(),3)
    return backend,records,holder


@pytest.mark.asyncio
async def test_same_persistent_thread_loads_three_real_pcm_warmups_and_closes(tmp_path):
    gate=threading.Event();backend,records,holder=await started(tmp_path,gate)
    try:
        await asyncio.sleep(.01)
        assert not backend.is_ready and backend.state=='loading'
        assert not backend.warmup
        gate.set();await asyncio.wait_for(backend.ready_event.wait(),3)
        assert backend.is_ready and len(backend.warmup)==3
        assert all(r['pcm_bytes']>0 for r in backend.warmup)
        async with aclosing(backend.stream('Teste curto.',asyncio.Event())) as stream:
            data=b''.join([block async for block in stream])
        expected=pcm16_bytes(resample_poly(holder[0].audio,1,2,window=('kaiser',10.)))
        assert data==expected and len(data)//2==5002
        assert holder[0].closed_streams==4
    finally:gate.set();await backend.close()
    assert len({thread for event,thread in records})==1
    assert records[-1][0]=='close'


@pytest.mark.asyncio
async def test_native_first_audio_is_available_before_eof_and_prefetch_is_bounded(tmp_path):
    backend,_,holder=await started(tmp_path)
    stream=backend.stream('many',asyncio.Event())
    try:
        first=await anext(stream)
        await asyncio.sleep(.03)
        assert first and len(holder[0].steps)<=4
        assert holder[0].closed_streams==3
    finally:await stream.aclose();await backend.close()
    assert holder[0].closed_streams==4


@pytest.mark.asyncio
async def test_cancel_while_producer_waits_for_buffer_closes_and_reset_precedes_next_call(tmp_path):
    backend,records,holder=await started(tmp_path);cancel=asyncio.Event();stream=backend.stream('many',cancel)
    try:
        await anext(stream);await asyncio.sleep(.03);cancel.set()
        assert [b async for b in stream]==[]
        stopped=len(holder[0].steps)
        await asyncio.sleep(.01);assert len(holder[0].steps)==stopped
        assert records[-1][0]=='reset'
        async with aclosing(backend.stream('Outra frase.',asyncio.Event())) as following:
            assert b''.join([b async for b in following])
        assert backend.is_ready
    finally:await stream.aclose();await backend.close()


@pytest.mark.asyncio
async def test_pre_cancel_does_not_run_any_additional_native_step(tmp_path):
    backend,records,_=await started(tmp_path);cancel=asyncio.Event();cancel.set();before=list(records)
    try:assert [b async for b in backend.stream('many',cancel)]==[];assert records==before
    finally:await backend.close()


@pytest.mark.asyncio
async def test_stalled_consumer_has_bounded_timeout_and_owned_cleanup(tmp_path):
    backend,records,_=await started(tmp_path,client_stall_seconds=.1);stream=backend.stream('many',asyncio.Event())
    try:
        await anext(stream);await asyncio.sleep(.2)
        with pytest.raises(TimeoutError,match='stalled'):await stream.aclose()
        assert records[-1][0]=='reset'
    finally:await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['empty','invalid'])
async def test_nonfinite_or_empty_audio_never_becomes_successful_pcm(tmp_path,text):
    backend,records,_=await started(tmp_path)
    try:
        with pytest.raises((ValueError,RuntimeError)):
            async with aclosing(backend.stream(text,asyncio.Event())) as stream:
                [b async for b in stream]
        assert records[-1][0]=='reset'
    finally:await backend.close()


@pytest.mark.asyncio
async def test_onset_stage_preserves_internal_pause_and_matches_existing_trimmer(tmp_path):
    backend,_,_=await started(tmp_path)
    try:
        pcm=b''.join([b async for b in backend.stream('silêncio',asyncio.Event())])
        onset=OutputOnsetTrimmer(24000)
        expected=b''.join(onset.push(pcm)+onset.finish())
        actual=b''.join([b async for b in processed_stream(backend,'silêncio',asyncio.Event(),onset_trim='conservative-v1')])
        assert actual==expected and len(actual)<len(pcm)
        assert actual.count(b'\0'*2000)>0
    finally:await backend.close()


def test_exact_capacity_guard_accepts_boundary_and_refuses_overflow():
    cfg=SimpleNamespace(max_len=400,capacity=1024)
    assert runtime.capacity_guard(622,cfg)['reserved_slots']==1024
    with pytest.raises(runtime.BackendTextTooLong):runtime.capacity_guard(623,cfg)


def test_original_frontend_preserves_aegis_digits_case_and_sentence_boundaries():
    original='Aegis, leia às 14h05. A CPU usa 24 kHz.'
    assert OriginalTextNormalizer().normalize(original).normalized_text==original
    assert OriginalTextNormalizer().normalize('pressa\u0303o').normalized_text=='pressão'
    for value in ['x\0y','x\ud800y']:
        with pytest.raises(ValueError):OriginalTextNormalizer().normalize(value)


def test_refinement_copy_is_hash_bound_and_equivalent_on_seven_validation_texts():
    from training.f5_refine_text import normalize_text as original
    from aegis_speech.experimental_refinement import normalize_text,F5_SOURCE_SHA256
    root=Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'training/f5_refine_text.py').read_bytes()).hexdigest()==F5_SOURCE_SHA256
    corpus=json.loads((root/'benchmarks/v2/validation.pt-BR.json').read_text())
    assert len(corpus)==7
    for row in corpus:
        assert normalize_text(row['text'])==original(row['text'])
        assert RefinedTextNormalizer().normalize(row['text']).normalized_text==original(row['text'])['tts_text']


@pytest.mark.parametrize('literal',['3.14','10.1.1.47','nome@example.org','https://example.org/teste'])
def test_original_technical_tokens_survive_before_a_later_sentence(literal):
    source='Leia '+literal+' agora. Depois feche a porta.'
    result=OriginalTextNormalizer().normalize(source)
    segments=OriginalTextSegmenter(min_chars=1,min_words=1).segment(result)
    assert len(segments)==2
    assert literal in segments[0].normalized_text
    assert ' '.join(r.normalized_text for r in segments)==source
    assert all('¤' not in r.normalized_text and '\0' not in r.normalized_text for r in segments)


def test_base_loader_refuses_missing_non_lora_even_when_upstream_uses_strict_false():
    class Model:
        def named_parameters(self):return [('weight',object()),('layer.lora_A',object())]
        def state_dict(self):return {'weight':0,'layer.lora_A':0}
        def load_state_dict(self,state,strict=False):
            return SimpleNamespace(missing_keys=set(self.state_dict())-set(state),unexpected_keys=set(state)-set(self.state_dict()))
    old=Model.load_state_dict
    with runtime.strict_base_load(Model):
        Model().load_state_dict({'weight':0},strict=False)
        with pytest.raises(ValueError):Model().load_state_dict({},strict=False)
        with pytest.raises(ValueError):Model().load_state_dict({'weight':0,'unexpected':0},strict=False)
    assert Model.load_state_dict is old


def test_model_and_gpu_lease_are_released_even_when_reset_raises(tmp_path):
    value=runtime.NativeSession(config(tmp_path));released=[]
    value.model=object();value.lease=SimpleNamespace(close=lambda:released.append('lease'))
    def broken():raise RuntimeError('simulated reset failure')
    value.reset=broken
    with pytest.raises(RuntimeError,match='reset failure'):value.close()
    assert value.model is None and value.lease is None and released==['lease']


def test_replay_counter_preserves_result_and_restores_original_on_failed_close(tmp_path):
    class Graph:
        def replay(self,value):return value+7
    original=Graph.replay
    value=runtime.NativeSession(config(tmp_path,instrument_replays=True))
    value.torch=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph,empty_cache=lambda:None))
    value.instrument_replays()
    assert Graph().replay(5)==12 and value.replays==1
    def broken():raise RuntimeError('fixture cleanup')
    value.reset=broken
    with pytest.raises(RuntimeError):value.close()
    assert Graph.replay is original


@pytest.mark.asyncio
async def test_executor_is_shutdown_even_if_model_close_raises(tmp_path):
    backend,_,holder=await started(tmp_path)
    def broken():raise RuntimeError('simulated close failure')
    holder[0].close=broken
    with pytest.raises(RuntimeError,match='close failure'):await backend.close()
    assert backend.state=='closed' and backend.session is None and backend.executor._shutdown


@pytest.mark.asyncio
async def test_cancelled_initialization_still_closes_model_on_its_owner_thread(tmp_path):
    backend=runtime.VoxBackend(config(tmp_path));closed=[]
    backend.session=SimpleNamespace(close=lambda:closed.append(threading.get_ident()))
    backend.initializing=asyncio.create_task(asyncio.sleep(100))
    backend.initializing.cancel();await asyncio.gather(backend.initializing,return_exceptions=True)
    with pytest.raises(asyncio.CancelledError):await backend.close()
    assert len(closed)==1 and closed[0]!=threading.get_ident()
    assert backend.session is None and backend.state=='closed' and backend.executor._shutdown


def test_runtime_resampler_has_the_same_phase_and_samples_as_the_validated_probe():
    from training.v2_voxcpm_streaming_probe import HalfRateFIR as Validated
    rng=np.random.default_rng(40);y=rng.normal(0,.1,10003).astype(np.float32)
    a,b=HalfRateFIR(),Validated();pieces_a=[];pieces_b=[];offset=0
    for size in (1,0,3,17,2,41,7,89,4095,3841,1907):
        chunk=y[offset:offset+size];offset+=len(chunk)
        pieces_a.append(a.push(chunk));pieces_b.append(b.push(chunk))
    assert offset==len(y)
    pieces_a.append(a.flush());pieces_b.append(b.flush())
    np.testing.assert_array_equal(np.concatenate(pieces_a),np.concatenate(pieces_b))
    assert a.recipe()==b.recipe()


@pytest.mark.asyncio
@pytest.mark.parametrize('frontend',['original-nfc','f5-refinement-1'])
async def test_gateway_contracts_readiness_profile_wav_pcm_and_preheader_overflow(tmp_path,monkeypatch,frontend):
    cfg=config(tmp_path,frontend=frontend);gate=threading.Event();records=[];holder=[]
    backend_class=runtime.VoxBackend
    def backend_factory(config):
        def session_factory(config):
            session=Session(config,records,gate);holder.append(session);return session
        return backend_class(config,session_factory)
    monkeypatch.setattr(gateway,'VOX_CONFIG',cfg)
    monkeypatch.setattr(gateway,'NATIVE_PROFILE',runtime.NATIVE_PROFILE)
    monkeypatch.setattr(gateway,'VOICE_PROFILE',cfg.profile)
    monkeypatch.setattr(runtime,'VoxBackend',backend_factory)
    monkeypatch.setitem(sys.modules,'voxcpm_runtime',runtime)
    monkeypatch.setattr(gateway,'_verify_protected_assets',lambda:pytest.fail('Vox must not read protected v1 reference'))
    old=dict(gateway.app.state._state)
    try:
        async with gateway.lifespan(gateway.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
                assert (await client.get('/health')).status_code==200
                assert (await client.get('/ready')).status_code==503
                cold=await client.post('/v1/audio/speech',json={'input':'Teste.'})
                assert cold.status_code==503 and cold.json()['detail']['code']=='BACKEND_NOT_READY'
                gate.set();await asyncio.wait_for(gateway.app.state.vox_backend.ready_event.wait(),3)
                assert (await client.get('/ready')).status_code==200
                wrong=await client.post('/v1/aegis/speech',json={'text':'Teste.','voice_profile':'AegisVoicev1.0'})
                assert wrong.status_code==422
                too_long=await client.post('/v1/audio/speech',json={'input':'overflow'})
                assert too_long.status_code==413 and too_long.json()['detail']['code']=='BACKEND_TEXT_TOO_LONG'
                wav=await client.post('/v1/audio/speech',json={'input':'Aegis, leia às 14h05.','response_format':'wav','stream':False})
                assert wav.status_code==200 and wav.headers['x-aegis-voice-profile']==cfg.profile
                assert wav.headers['x-aegis-tts-dsp-variant']=='passthrough'
                assert wav.headers['x-aegis-tts-frontend']==frontend
                normalizer=OriginalTextNormalizer() if frontend=='original-nfc' else RefinedTextNormalizer()
                assert holder[0].inputs[-1]==normalizer.normalize('Aegis, leia às 14h05.').normalized_text
                with wave.open(io.BytesIO(wav.content)) as audio:
                    assert (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getnframes())==(1,2,24000,5002)
                    expected=audio.readframes(audio.getnframes())
                pcm=await client.post('/v1/aegis/speech',json={'text':'Teste.','response_format':'pcm','stream':True})
                assert pcm.status_code==200 and pcm.content==expected
                assert pcm.headers['x-aegis-audio-format']=='pcm_s16le'
                bad=await client.post('/v1/aegis/speech',json={'text':'Teste.','stream':True,'response_format':'wav'})
                assert bad.status_code==422
                identifier=pcm.headers['x-aegis-request-id']
                status=await client.get('/v1/aegis/speech/'+identifier)
                assert status.json()['status']=='completed'
                holder[0].generation_gate=threading.Event()
                identifier='0'*26
                response=asyncio.create_task(client.post('/v1/aegis/speech',json={'request_id':identifier,'text':'many',
                    'response_format':'pcm','stream':True}))
                for _ in range(100):
                    if len(holder[0].steps)>=2:break
                    await asyncio.sleep(.005)
                cancelled=await client.delete('/v1/aegis/speech/'+identifier)
                assert cancelled.status_code==200 and cancelled.json()['status'] in {'cancelling','cancelled'}
                holder[0].generation_gate.set()
                await asyncio.wait_for(response,3)
                status=await client.get('/v1/aegis/speech/'+identifier)
                assert status.json()['status']=='cancelled'
                assert (await client.get('/metrics')).status_code==200
    finally:
        gate.set();gateway.app.state._state.clear();gateway.app.state._state.update(old)
