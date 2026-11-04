"""Future V4 ledger admission, mocks only; no checkpoint/torch/Docker/GPU."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from training import v2_fastpitch_budget_gate_v4 as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'gate';self.run.mkdir()
        self.baseline=self.root/'baseline.json';gate.protocol.save(self.baseline,{})
        self.ledger=self.root/'external.json';gate.protocol.save(self.ledger,{})
        self.args=SimpleNamespace(run=self.run,cache_baseline=self.baseline,external_ledger=self.ledger,gpu_authorized=True,review=self.root/'review.json')
        self.plan=dict(command=['/synthetic/resume'],source_hashes={});gate.protocol.save(self.run/'plan.json',self.plan)
        self.owner=dict(self=dict(pid=100,start_ticks='1'),worker=dict(pid=90,start_ticks='2'),guard_owner=dict(pid=80,start_ticks='3'))
        self.active=dict(status='active',owner_authorized=True,production_stopped=True,exclusive_aegisvoice_v2_gpu=True,container_id=gate.qwen.CONTAINER)
        gate.protocol.save(self.root/'gpu-window-authorized.json',self.active)
        self.v3=SimpleNamespace(worker_guard=lambda:(90,'2'),gpu_empty=lambda:None,
            run_gate=lambda *args,**kwargs:dict(inventory=dict(allocated_bytes=1_000_000_000,permission_errors=[]),checkpoint_files=[]),
            plan_budget=lambda *args:dict(allowed=True,projected_peak_bytes=3_000_000_000,extra_peak_bytes=1_000_000_000),
            LIMIT=40_000_000_000,FREE_RESERVE=5*1024**3)

    def mocks(self,budget=None):
        if budget is None:
            def budget(*args,**kwargs):
                self.assertEqual(gate.protocol.read(self.root/'gpu-window-authorized.json')['status'],'budget_reconciliation_required')
                return dict(total_accounted_bytes=1_100_000_000,free_bytes=20_000_000_000)
        return [patch.object(gate,'ROOT',self.root),patch.object(gate.qwen,'ROOT',self.root),
            patch.object(gate.qwen,'CACHE_BASELINE',self.baseline),patch.object(gate,'review_guard',return_value={}),
            patch.object(gate,'load_v3',return_value=self.v3),patch.object(gate.qwen,'owner_identity',return_value=self.owner),
            patch.object(gate.qwen,'same_owner',return_value=True),patch.object(gate.qwen,'original',return_value=dict(State=dict(Running=False))),
            patch.object(gate.qwen,'phase_budget',side_effect=budget),
            patch.dict(os.environ,dict(CUDA_CACHE_PATH=str(gate.CUDA_CACHE),CUDA_CACHE_MAXSIZE=str(gate.CUDA_MAX)))]

    def test_fresh_gate_closes_before_audit_and_consumes_ledger_before_active(self):
        with ExitStack() as stack:
            for mock in self.mocks():stack.enter_context(mock)
            result=gate.evaluate(self.args,self.plan)
        value=gate.protocol.read(self.root/'gpu-window-authorized.json')
        self.assertTrue(result['allowed']);self.assertEqual(value['status'],'active')
        self.assertEqual(result['six_state_plan']['cuda_cache_future_reserve_bytes'],18*1024**2)
        consumed=value['v4_ledger_consumed'];self.assertEqual(consumed['external_specification_sha256'],gate.protocol.sha(self.ledger))
        self.assertEqual(consumed['sha256'],gate.protocol.sha(self.run/'reconciliation.json'))

    def test_refused_budget_never_reopens_active(self):
        self.v3.plan_budget=lambda *args:dict(allowed=False,projected_peak_bytes=41_000_000_000,extra_peak_bytes=1_000_000_000)
        with ExitStack() as stack:
            for mock in self.mocks():stack.enter_context(mock)
            with self.assertRaisesRegex(RuntimeError,'budget refused'):gate.evaluate(self.args,self.plan)
        value=gate.protocol.read(self.root/'gpu-window-authorized.json')
        self.assertEqual(value['status'],'budget_reconciliation_required');self.assertFalse(value['exclusive_aegisvoice_v2_gpu'])

    def test_changed_authorization_is_not_overwritten_at_admission(self):
        def changed(*args,**kwargs):
            gate.protocol.save(self.root/'gpu-window-authorized.json',dict(status='emergency_closed'))
            return dict(total_accounted_bytes=1_100_000_000,free_bytes=20_000_000_000)
        with ExitStack() as stack:
            for mock in self.mocks(changed):stack.enter_context(mock)
            with self.assertRaisesRegex(RuntimeError,'changed'):gate.evaluate(self.args,self.plan)
        self.assertEqual(gate.protocol.read(self.root/'gpu-window-authorized.json')['status'],'emergency_closed')

    def test_stopped_owner_and_exclusivity_flags_are_mandatory(self):
        for key in ('owner_authorized','production_stopped','exclusive_aegisvoice_v2_gpu'):
            gate.protocol.save(self.root/'gpu-window-authorized.json',dict(self.active,**{key:False}))
            with patch.object(gate,'ROOT',self.root),self.assertRaises(RuntimeError):gate.authorization(self.owner,self.baseline)

    def test_pending_qwen_requires_bound_supplement_and_complete_cleanup(self):
        preparation=self.root/'qwen';run=preparation/'validation';run.mkdir(parents=True)
        supplement=run/'resume-budget-supplement.json';gate.protocol.save(supplement,{})
        gate.protocol.save(run/'plan.json',{})
        gate.protocol.save(run/'execution.json',dict(status='failed_requires_root_review',cleanup_error=None,sources_unchanged=True,
            cleanup=dict(qwen_evidence_complete=True,stopped=True,gpu_empty=True)))
        pending=dict(self.active,status='budget_reconciliation_required',exclusive_aegisvoice_v2_gpu=False,
            qwen_baseline=dict(worker_pid=90,owner_pid=80),budget_reconciliation=dict(run=str(run),qwen_evidence_complete=True,
                executor_sha256=gate.protocol.sha(gate.qwen.__file__),
                cache_baseline=str(self.baseline),cache_baseline_sha256=gate.protocol.sha(self.baseline),
                plan_sha256=gate.protocol.sha(run/'plan.json'),supplement_path=str(supplement),supplement_sha256=gate.protocol.sha(supplement),
                original_experimental_authorization=self.active))
        gate.protocol.save(self.root/'gpu-window-authorized.json',pending)
        with patch.object(gate,'ROOT',self.root),patch.object(gate.qwen,'PREPARATION',preparation):
            _,consumed=gate.authorization(self.owner,self.baseline)
            self.assertEqual(consumed['qwen_measurement_status'],'failed_requires_root_review')
            gate.protocol.save(supplement,dict(changed=True))
            with self.assertRaisesRegex(RuntimeError,'supplement'):gate.authorization(self.owner,self.baseline)


if __name__=='__main__':unittest.main()
