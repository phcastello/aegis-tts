from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import glados_optimization as optimization


def weights():
    return {"pitch_centralization":.2,"pitch_modulation":.2,"formant_shift":.2,"chains":.4}


def test_normalized_distance_anchors_and_non_informative_exclusion():
    assert optimization.normalized_distance([2],[2],[0],minimum=.02)["score"]==0
    assert optimization.normalized_distance([0],[2],[0],minimum=.02)["score"]==1
    weak=optimization.normalized_distance([1],[.01],[0],minimum=.02,epsilon=1e-8)
    assert weak["status"]=="non_informative" and weak["score"] is None


def test_effect_context_aggregation_renormalizes():
    contexts=[{"status":"informative","score":0.0},{"status":"non_informative","score":None},
              {"status":"informative","score":1.0}]
    score=optimization.aggregate_effect_contexts(contexts)
    assert score["score"]==pytest.approx(.75) and score["informative_contexts"]==2


def test_informative_matrix_blocks_fewer_than_two_contexts():
    features={name:{effect:[0.] for effect in optimization.EFFECTS} for name in ("none","C","M","F","CM","FM","CF","CFM")}
    config={"scoring":{"denominator_epsilon":1e-8,"minimum_informative_distance":{"pitch":.02,"modulation":.02,"formant":.02}}}
    matrix=optimization.informative_matrix(features,config)
    assert not matrix["ready"] and all(value==0 for value in matrix["informative_context_counts"].values())


def test_artifact_allowance_uses_worst_reference_plus_tolerance():
    value=optimization.artifact_excess({"clicks":8},{"clicks":3},{"clicks":5},{"clicks":1},{"clicks":.01})
    assert value["details"]["clicks"]=={"allowed":6.,"output":8.,"excess":2.,"weight":.01,"penalty":.02}
    assert value["total"]==pytest.approx(.02)


def test_chain_record_score_includes_artifacts_consistently():
    record={"objectives":[.5,1.,1.5],"artifacts":{"total":.1}}
    assert optimization.chain_record_score(record)==pytest.approx(1.5)
    record["minimax"]=1.7
    assert optimization.chain_record_score(record)==pytest.approx(1.7)


def test_final_score_and_constraints():
    values=dict(zip(optimization.EFFECTS,[.5,1.,1.5]))
    assert optimization.final_chain_score(values,.1)==pytest.approx(1.5)
    assert any(x>0 for x in optimization.technical_constraints({"empty":True,"duration":0},expected_duration=10))
    assert len(optimization.technical_constraints({"duration":10},expected_duration=10))==optimization.CONSTRAINT_COUNT
    assert optimization.merge_constraints([[0,0,0,0,0,0],[0,1,0,0,0,0]])==[0,1,0,0,0,0]
    with pytest.raises(ValueError):optimization.merge_constraints([[1]])


def test_allocation_checkpoint_and_generations():
    assert optimization.allocation(200,weights())=={"pitch_centralization":40,"pitch_modulation":40,"formant_shift":40,"chains":80}
    assert [trial//20 for trial in range(80)].count(3)==20


def test_feasible_trials_dominate_infeasible_front():
    feasible={"trial":1,"objectives":[1.,1.,1.],"constraints":[0.]}
    infeasible={"trial":2,"objectives":[0.,0.,0.],"constraints":[1.]}
    assert optimization.pareto_front([feasible,infeasible])==[feasible]


def test_hypervolume_uses_fixed_reference():
    assert optimization.hypervolume_3d([[1,1,1]],[2,2,2])==1
    assert optimization.hypervolume_3d([[3,1,1]],[2,2,2])==0


def test_lineage_is_never_inferred(tmp_path):
    record={"generation":0,"individuals":[],"elite_survivors":[],"lineage_available":False,
            "parents":None,"crossover":None,"mutations":None}
    optimization.atomic_text(tmp_path/"populations.jsonl",json.dumps(record)+"\n")
    loaded=optimization.read_jsonl(tmp_path/"populations.jsonl")[0]
    assert loaded["lineage_available"] is False and loaded["parents"] is None


def test_config_pins_checkpoint_contract():
    config=optimization.load_config(ROOT/"config/glados-effect-references.yaml")
    assert config["optimization"]["tpe"]=={"seed":40733133,"n_startup_trials":10,"multivariate":True,"group":True}
    assert config["optimization"]["chains"]["population_size"]==20
    assert config["validation"]["participate_in_optimization"] is False
    assert config["scoring"]["schema_version"]==optimization.SCORING_SCHEMA_VERSION


def test_dashboard_can_be_rebuilt_from_jsonl(tmp_path):
    env={"AEGIS_GLADOS_RE_OUTPUT_DIR":str(tmp_path)};out=tmp_path/"optimization";out.mkdir()
    optimization.append_jsonl(out/"trials.jsonl",{"study":"chains","trial":0,"objectives":[1,1,1],"constraints":[],"feasible":True,"audio_file":"audio/x.wav"})
    target=optimization.rebuild_dashboard(env)
    assert target.is_file() and "lineage_available=false" in target.read_text()


def test_nsga_second_generation_accepts_fixed_constraints():
    optuna=pytest.importorskip("optuna")
    config=optimization.load_config(ROOT/"config/glados-effect-references.yaml")
    study=optuna.create_study(directions=["minimize"]*3,sampler=optimization._sampler(config,"chains"))
    def objective(trial):
        x=trial.suggest_float("x",0,1)
        trial.set_user_attr("constraints",[0,0,0,0,0,0] if trial.number%2 else [0,0,.1,0,0,0])
        return x,1-x,abs(.5-x)
    study.optimize(objective,n_trials=40)
    assert len(study.trials)==40 and all(t.state.name=="COMPLETE" for t in study.trials)


def test_resume_rejects_study_without_pipeline_schema(tmp_path):
    optuna=pytest.importorskip("optuna")
    config=optimization.load_config(ROOT/"config/glados-effect-references.yaml")
    storage=f"sqlite:///{tmp_path/'studies.sqlite3'}"
    legacy=optuna.create_study(study_name="chains",storage=storage,directions=["minimize"]*3)
    legacy.enqueue_trial({"x":.5})
    legacy.optimize(lambda trial:(trial.suggest_float("x",0,1),1.,1.),n_trials=1)
    with pytest.raises(ValueError,match="predates pipeline schema"):
        optimization._create_study(config,tmp_path,"chains",resume=True)


def test_resume_advances_sampler_seed_and_does_not_repeat_first_suggestion(tmp_path):
    pytest.importorskip("optuna")
    config=optimization.load_config(ROOT/"config/glados-effect-references.yaml")
    first=optimization._create_study(config,tmp_path,"pitch_centralization",resume=False)
    first.optimize(lambda trial:trial.suggest_float("x",0,1),n_trials=1)
    resumed=optimization._create_study(config,tmp_path,"pitch_centralization",resume=True)
    resumed.optimize(lambda trial:trial.suggest_float("x",0,1),n_trials=1)
    assert resumed.trials[0].params["x"]!=resumed.trials[1].params["x"]
    assert resumed.user_attrs["sampler_seed"]==config["optimization"]["tpe"]["seed"]+1


def test_invalid_run_is_archived_without_deleting_cache(tmp_path):
    env={"AEGIS_GLADOS_RE_OUTPUT_DIR":str(tmp_path)};out=tmp_path/"optimization";(out/"cache").mkdir(parents=True)
    (out/"studies.sqlite3").write_bytes(b"sqlite");(out/"cache"/"x.wav").write_bytes(b"wav")
    result=optimization.archive_invalid_optimization(env,"test")
    assert result["history_preserved"] and (tmp_path/"optimization-invalid-checkpoint-001"/"studies.sqlite3").is_file()
    assert (tmp_path/"optimization"/"cache"/"x.wav").read_bytes()==b"wav"
