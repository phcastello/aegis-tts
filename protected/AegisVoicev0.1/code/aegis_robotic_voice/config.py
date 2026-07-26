from __future__ import annotations
import os, re
from dataclasses import dataclass
from pathlib import Path
import yaml

# config.py -> package -> src -> module root
ROOT = Path(__file__).resolve().parents[2]
KNOWN = {"AEGIS_ROBOTIC_VOICE_ROOT", "AEGIS_CHATTERBOX_MODEL", "AEGIS_CHATTERBOX_REVISION", "AEGIS_CHATTERBOX_DEVICE", "AEGIS_CHATTERBOX_DTYPE", "AEGIS_CHATTERBOX_LANGUAGE_ID", "AEGIS_CHATTERBOX_SEED", "AEGIS_CHATTERBOX_ALLOW_DOWNLOADS", "AEGIS_CHATTERBOX_USE_COMPILE", "AEGIS_CHATTERBOX_USE_CPU_OFFLOAD", "AEGIS_AEGIS_GLADOS_REFERENCE_WAV", "AEGIS_CHATTERBOX_NEUTRAL_REFERENCE_WAV", "AEGIS_ROBOTIC_VOICE_RESULTS_DIR", "AEGIS_ROBOTIC_VOICE_REPORTS_DIR", "AEGIS_ROBOTIC_VOICE_CACHE_DIR", "AEGIS_ROBOTIC_VOICE_MODELS_DIR", "AEGIS_ROBOTIC_VOICE_ENVS_DIR", "AEGIS_ROBOTIC_VOICE_DSP_WORKERS", "AEGIS_ROBOTIC_VOICE_FRAME_PERIOD_MS", "AEGIS_ROBOTIC_VOICE_LISTENING_LUFS", "AEGIS_ROBOTIC_VOICE_LISTENING_TRUE_PEAK_DB", "AEGIS_ROBOTIC_VOICE_LOG_LEVEL", "AEGIS_ROBOTIC_VOICE_COMMAND_TIMEOUT_SECONDS", "AEGIS_QWEN_COLD_REFERENCE_WAV", "AEGIS_QWEN_COLD_REFERENCE_TEXT", "AEGIS_QWEN_COLD_PORT"}
LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")

def load_dotenv(path: Path) -> dict[str, str]:
    """Strict dotenv reader: known literal KEY=value only; never shell-expands."""
    out = {}
    if not path.exists(): return out
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"): continue
        match = LINE.fullmatch(line)
        if not match or match.group(1) not in KNOWN or "$" in match.group(2) or "`" in match.group(2):
            raise ValueError(f"invalid .env line {number}")
        out[match.group(1)] = match.group(2)
    return out

@dataclass(frozen=True)
class Settings:
    root: Path; results: Path; reports: Path; cache: Path; models: Path; envs: Path
    model: str; revision: str; device: str; dtype: str; language_id: str; seed: int; frame_period_ms: float

def settings(overrides: dict[str, str] | None = None) -> Settings:
    values = {"AEGIS_ROBOTIC_VOICE_ROOT": str(ROOT), "AEGIS_CHATTERBOX_MODEL": "ResembleAI/Chatterbox-Multilingual-pt-br", "AEGIS_CHATTERBOX_REVISION": "b3952f18bc2eaa72b9bd7c17d2c4653bcad4770d", "AEGIS_CHATTERBOX_DEVICE": "cuda", "AEGIS_CHATTERBOX_DTYPE": "float32", "AEGIS_CHATTERBOX_LANGUAGE_ID": "pt", "AEGIS_CHATTERBOX_SEED": "40733133", "AEGIS_ROBOTIC_VOICE_FRAME_PERIOD_MS": "5.0"}
    values.update(load_dotenv(ROOT / ".env")); values.update({k:v for k,v in os.environ.items() if k in KNOWN}); values.update(overrides or {})
    root = Path(values["AEGIS_ROBOTIC_VOICE_ROOT"])
    def path(key, default): return Path(values.get(key, str(root / default)))
    return Settings(root, path("AEGIS_ROBOTIC_VOICE_RESULTS_DIR","results"), path("AEGIS_ROBOTIC_VOICE_REPORTS_DIR","reports/generated"), path("AEGIS_ROBOTIC_VOICE_CACHE_DIR","private/cache"), path("AEGIS_ROBOTIC_VOICE_MODELS_DIR","private/models"), path("AEGIS_ROBOTIC_VOICE_ENVS_DIR","private/environments"), values["AEGIS_CHATTERBOX_MODEL"], values["AEGIS_CHATTERBOX_REVISION"], values["AEGIS_CHATTERBOX_DEVICE"], values["AEGIS_CHATTERBOX_DTYPE"], values["AEGIS_CHATTERBOX_LANGUAGE_ID"], int(values["AEGIS_CHATTERBOX_SEED"]), float(values["AEGIS_ROBOTIC_VOICE_FRAME_PERIOD_MS"]))

def yaml_config(name: str):
    with (ROOT / "config" / name).open(encoding="utf-8") as file: return yaml.safe_load(file)
