"""Explicit regional Chatterbox loader, derived from the official pt-BR Space revision."""
from __future__ import annotations
import hashlib, os, shutil
from pathlib import Path
import numpy as np
from ..config import Settings, yaml_config

REGIONAL = "ResembleAI/Chatterbox-Multilingual-pt-br"
T3_FILE = "t3_pt_br.safetensors"; S3_FILE = "s3gen_v3.safetensors"; TOKENIZER = "grapheme_mtl_merged_expanded_v1.json"

def validate_checkpoint_identity(metadata: dict) -> None:
    expected={"model":REGIONAL,"locale":"pt-BR","language_id":"pt","decoder":"V3","tokenizer":TOKENIZER,"t3":T3_FILE,"s3gen":S3_FILE}
    for key,value in expected.items():
        if metadata.get(key)!=value: raise ValueError(f"incompatible Chatterbox {key}: {metadata.get(key)!r}, expected {value!r}")

def _digest(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def prepare_assets(settings: Settings) -> tuple[Path, dict]:
    """Download only fixed-revision files; verify every available SHA-256."""
    spec=yaml_config("chatterbox.yaml"); validate_checkpoint_identity({"model":settings.model,"locale":spec["locale"],"language_id":settings.language_id,"decoder":spec["decoder"],"tokenizer":spec["tokenizer"],"t3":T3_FILE,"s3gen":S3_FILE})
    target=settings.models / "chatterbox-ptbr" / settings.revision; target.mkdir(parents=True,exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e: raise RuntimeError("install huggingface_hub in the Chatterbox environment") from e
    if os.environ.get("AEGIS_CHATTERBOX_ALLOW_DOWNLOADS", "1") != "1" and not all((target/n).exists() for n in [T3_FILE,S3_FILE,TOKENIZER,"ve.pt","conds.pt"]): raise RuntimeError("assets missing and downloads disabled")
    requests=[(settings.model,settings.revision,T3_FILE),(settings.model,settings.revision,S3_FILE),(settings.model,settings.revision,TOKENIZER),(spec["base_repository"],spec["base_revision"],"ve.pt"),(spec["base_repository"],spec["base_revision"],"conds.pt")]
    for repo,revision,name in requests:
        dest=target/name
        if not dest.exists(): shutil.copy2(hf_hub_download(repo_id=repo,revision=revision,filename=name),dest)
    for name, expected in spec["assets"].items():
        actual=_digest(target/name)
        if actual != expected: raise RuntimeError(f"hash mismatch for {name}: {actual}")
    return target, {"model":settings.model,"revision":settings.revision,"locale":"pt-BR","language_id":"pt","decoder":"V3","tokenizer":TOKENIZER,"t3":T3_FILE,"s3gen":S3_FILE,"assets":spec["assets"]}

class RegionalChatterbox:
    def __init__(self, model, sample_rate, identity): self.model=model; self.sample_rate=sample_rate; self.identity=identity; self._conditions={}
    @classmethod
    def load(cls, settings: Settings):
        import gc, torch
        from safetensors.torch import load_file
        from chatterbox.models.t3 import T3
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.tokenizers import MTLTokenizer
        from chatterbox.models.voice_encoder import VoiceEncoder
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
        directory,identity=prepare_assets(settings); device=settings.device
        # The host has 8 GiB RAM while the T3 safetensors file alone is 2.14 GB.
        # Allocate each module, move it to CUDA, load its state directly to CUDA,
        # then immediately release the temporary state dict. This is not offload:
        # all inference modules stay resident on the GPU after loading.
        target_device=device if str(device).startswith("cuda") else "cpu"
        ve=VoiceEncoder(); ve.load_state_dict(torch.load(directory/"ve.pt",map_location="cpu",weights_only=True)); ve.to(device).eval()
        t3=T3(T3Config.multilingual()).to(device).eval()
        t3_state=load_file(directory/T3_FILE, device=target_device); t3.load_state_dict(t3_state); del t3_state; gc.collect()
        # glibc may retain the temporary CPU allocator arenas after the T3
        # construction. Return them before S3Gen's constructor on 8 GiB hosts.
        try:
            import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        if target_device == "cuda": torch.cuda.synchronize(); torch.cuda.empty_cache()
        # S3Gen's normal constructor peaks above the remaining host RAM. Build
        # parameter shells on meta, then assign the safetensor storage on CUDA.
        with torch.device("meta"):
            s3=S3Gen()
        s3_state=load_file(directory/S3_FILE, device=target_device); s3.load_state_dict(s3_state,strict=False,assign=True); del s3_state; gc.collect()
        # These non-persistent analysis buffers are intentionally absent from
        # the checkpoint, so materialize the official constructor values here.
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.s3gen.const import S3GEN_SR
        s3.tokenizer._mel_filters=torch.from_numpy(librosa.filters.mel(sr=S3_SR,n_fft=s3.tokenizer.n_fft,n_mels=80)).float().to(device)
        s3.tokenizer.window=torch.hann_window(s3.tokenizer.n_fft,device=device)
        n_trim=S3GEN_SR // 50; trim=torch.zeros(2*n_trim,device=device); trim[n_trim:] = (torch.cos(torch.linspace(torch.pi,0,n_trim,device=device))+1)/2; s3.trim_fade=trim
        # Relative-position caches are regular attributes rather than buffers,
        # hence they are not covered by state_dict or named_buffers.
        for module in s3.modules():
            pe=getattr(module,"pe",None)
            if torch.is_tensor(pe) and pe.is_meta and hasattr(module,"extend_pe"):
                module.pe=None
                module.extend_pe(torch.zeros(1,5000,device=device))
        s3.eval()
        if target_device == "cuda": torch.cuda.synchronize(); torch.cuda.empty_cache()
        tokenizer=MTLTokenizer(str(directory/TOKENIZER)); conds=Conditionals.load(directory/"conds.pt",map_location="cpu").to(device)
        return cls(ChatterboxMultilingualTTS(t3,s3,ve,tokenizer,device,conds=conds), 24000, identity)
    def prepare_conditionals(self, prompt: str | None, exaggeration: float):
        key=(prompt,exaggeration)
        if prompt and key not in self._conditions:
            self.model.prepare_conditionals(prompt,exaggeration=exaggeration); self._conditions[key]=self.model.conds
        elif prompt: self.model.conds=self._conditions[key]
    def generate(self,text,*,audio_prompt=None,exaggeration=.2,cfg_weight=.3,temperature=.8,seed=40733133):
        import random, torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        self.prepare_conditionals(audio_prompt,exaggeration)
        with torch.inference_mode(): return self.model.generate(text,language_id="pt",audio_prompt_path=None,exaggeration=exaggeration,cfg_weight=cfg_weight,temperature=temperature)
