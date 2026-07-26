from __future__ import annotations
import json
from pathlib import Path

def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def latest_candidates(items: list[dict]) -> list[dict]:
    """Keep only the latest attempt for each logical candidate after reruns."""
    latest={}
    for item in items:
        key=(item.get("source_profile"),item.get("engine"),item.get("preset"),item.get("sentence_id"))
        latest[key]=item
    return list(latest.values())
