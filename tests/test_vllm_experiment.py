import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vllm-omni"


def test_experimental_compose_is_isolated_and_loopback_only():
    compose = yaml.safe_load((EXPERIMENT / "docker-compose.yml").read_text())
    assert set(compose["services"]) == {"aegis-tts-vllm"}
    service = compose["services"]["aegis-tts-vllm"]
    assert service["container_name"] == "aegis-tts-vllm"
    assert service["ports"] == ["127.0.0.1:8004:8091"]
    assert "aegis-tts" not in compose["services"]


def test_model_voice_and_cache_mounts_are_safe():
    compose = yaml.safe_load((EXPERIMENT / "docker-compose.yml").read_text())
    volumes = compose["services"]["aegis-tts-vllm"]["volumes"]
    assert "../../models/Qwen3-TTS-12Hz-1.7B-Base:/models/Qwen3-TTS-12Hz-1.7B-Base:ro" in volumes
    assert "../../voices/giseli-34s:/voices/giseli-34s:ro" in volumes
    assert "../../cache/vllm-omni:/cache" in volumes
    assert all("../../cache:/cache" != volume for volume in volumes)


def test_official_two_stage_optimized_structure_is_preserved():
    config = yaml.safe_load((EXPERIMENT / "qwen3-tts-a2000.yaml").read_text())
    assert config["async_chunk"] is True
    assert len(config["stages"]) == 2
    assert [stage["stage_id"] for stage in config["stages"]] == [0, 1]
    assert all(stage["devices"] == "0" for stage in config["stages"])
    assert all(stage["max_num_seqs"] == 1 for stage in config["stages"])
    assert config["stages"][0].get("enforce_eager") is not True
    assert config["stages"][1]["enforce_eager"] is False
    assert config["stages"][0]["async_scheduling"] is True
    assert config["stages"][1]["async_scheduling"] is True
    connector = config["connectors"]["connector_of_shared_memory"]
    assert connector["name"] == "SharedMemoryConnector"
    assert connector["extra"]["codec_streaming"] is True
    assert config["stages"][1]["max_num_batched_tokens"] == 65536
    assert config["stages"][1]["max_model_len"] == 65536


def test_benchmark_preset_is_exact():
    cfg = json.loads((EXPERIMENT / "benchmark-config.json").read_text())
    assert cfg["model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert cfg["voice"] == "giseli-34s"
    assert cfg["text"] == (
        "Olá, Pedro. Pelo que eu verifiquei, todos os serviços estão funcionando normalmente. "
        "A reunião continua marcada para segunda-feira, e ainda há algumas mensagens que precisam "
        "de resposta. Depois disso, posso organizar as próximas tarefas e revisar o que ficou "
        "pendente para amanhã."
    )
    assert cfg["task_type"] == "Base"
    assert cfg["language"] == "Portuguese"
    assert cfg["x_vector_only_mode"] is False
    assert cfg["seed"] == 41
    assert cfg["temperature"] == 0.8
    assert cfg["top_p"] == 0.9
    assert cfg["top_k"] == 50
    assert cfg["max_tokens"] == 1024
    assert cfg["min_tokens"] == 2
    assert cfg["repetition_penalty"] == 1.05
    assert cfg["do_sample"] is True


def test_stage_zero_sampling_matches_preset():
    config = yaml.safe_load((EXPERIMENT / "qwen3-tts-a2000.yaml").read_text())
    sampling = config["stages"][0]["default_sampling_params"]
    assert sampling == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "min_tokens": 2,
        "max_tokens": 1024,
        "seed": 41,
        "repetition_penalty": 1.05,
    }


def test_restore_trap_is_present_and_scoped():
    script = (ROOT / "scripts/vllm-benchmark.sh").read_text()
    assert "trap restore EXIT INT TERM" in script
    assert "vllm-stop.sh" in script
    assert "vllm-restore-production.sh" in script
    assert "docker stop aegis-tts" in script
    assert "docker compose down" not in script
    assert "docker system prune" not in script
    assert "docker volume prune" not in script


def test_reports_do_not_store_benchmark_or_reference_transcription():
    benchmark_text = json.loads((EXPERIMENT / "benchmark-config.json").read_text())["text"]
    reference_text = (ROOT / "voices/giseli-34s/reference.txt").read_text().strip()
    for path in (ROOT / "reports/vllm").glob("*"):
        if path.is_file() and path.suffix in {".md", ".json", ".txt", ".html", ".log", ".csv"}:
            content = path.read_text(errors="replace")
            assert benchmark_text not in content
            assert reference_text not in content
