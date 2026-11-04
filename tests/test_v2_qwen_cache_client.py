"""Owned CPU auditor contracts, synthetic pipes/config only; no Docker."""
import copy
import io
import json
import os
from pathlib import Path
import selectors
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_qwen_cache_client as client


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.audit=client.Auditor(self.root/'audit',lambda:None,'hash')
        self.identifier='a'*64
        self.frame=dict(schema=client.reader.SCHEMA,complete=True,root='/input-cache',cutoff_unix_ns=client.reader.GOAL_START_NS,
                        hashes_requested=True,entries={},physical_bytes=0,fd_anchored=True,metadata_roster_passes=2)

    def configuration(self):
        a=self.audit
        return dict(Id=self.identifier,Image=client.reader.IMAGE,Config=dict(User='0:0',Entrypoint=['python3'],
            Cmd=['-I','-S','-B','-u','/audit/reader.py','--reader-mode','hash','--root','/input-cache'],
            Env=['NVIDIA_VISIBLE_DEVICES=void','CUDA_VISIBLE_DEVICES='],Labels={
                'aegis.root.audit-name':a.name,'aegis.root.source-sha256':a.plan['source_sha256'],
                'aegis.root.audit':'qwen-cache-goal-accounting'}),HostConfig=dict(Runtime='runc',NetworkMode='none',
                ReadonlyRootfs=True,Privileged=False,CapAdd=['DAC_READ_SEARCH'],CapDrop=['ALL'],Devices=[],DeviceRequests=[],
                PortBindings={},LogConfig=dict(Type='none'),Memory=512*1024**2,NanoCpus=500_000_000,PidsLimit=32,
                SecurityOpt=['no-new-privileges:true']),Mounts=[dict(Type='bind',Source=client.reader.HOST_CACHE,Destination='/input-cache',RW=False),
                dict(Type='bind',Source=str(Path(client.reader.__file__).resolve()),Destination='/audit/reader.py',RW=False)])

    def process(self, payload):
        pipes=[]
        for data in (payload,b''):
            incoming,outgoing=os.pipe();os.write(outgoing,data);os.close(outgoing)
            pipes.append(os.fdopen(incoming,'rb',buffering=0))
        return SimpleNamespace(stdin=io.BytesIO(),stdout=pipes[0],stderr=pipes[1],wait=lambda timeout:0)

    def test_mock_full_hash_auditor_creates_validates_reads_and_removes_only_exact_id(self):
        events=[]
        def command(argv,**kwargs):
            events.append((argv,kwargs));return SimpleNamespace(stdout=self.identifier+'\n')
        proc=self.process(client.reader.encode_frame(self.frame))
        with patch.object(client.subprocess,'run',side_effect=command),patch.object(client.subprocess,'check_output',return_value=json.dumps([self.configuration()])),patch.object(client.subprocess,'Popen',return_value=proc):
            with self.audit as audit:self.assertEqual(audit.snapshot(),self.frame)
        self.assertEqual([item[0][1] for item in events],['create','stop','rm'])
        self.assertEqual(events[-2][0],['docker','stop','--time','1',self.identifier])
        self.assertEqual(events[-1][0],['docker','rm','-v',self.identifier])
        self.assertTrue(json.loads((self.audit.run/'receipt.json').read_text())['removed'])

    def test_original_id_can_never_be_a_start_stop_or_remove_target(self):
        with patch.object(client.subprocess,'run',return_value=SimpleNamespace(stdout=client.ORIGINAL)) as command,patch.object(client.subprocess,'Popen') as start:
            with self.assertRaises(RuntimeError):self.audit.__enter__()
        start.assert_not_called();self.assertEqual(command.call_count,1)

    def test_bad_runtime_or_writable_mount_never_starts_unvalidated_container(self):
        for change in ('runtime','mount'):
            audit=client.Auditor(self.root/change,lambda:None)
            self.audit=audit;value=self.configuration()
            if change=='runtime':value['HostConfig']['Runtime']='nvidia'
            else:value['Mounts'][0]['RW']=True
            with patch.object(client.subprocess,'run',return_value=SimpleNamespace(stdout=self.identifier)),patch.object(client.subprocess,'check_output',return_value=json.dumps([value])),patch.object(client.subprocess,'Popen') as start:
                with self.assertRaises(RuntimeError):audit.__enter__()
            start.assert_not_called()

    def test_partial_eof_frame_and_nonzero_hash_exit_are_rejected(self):
        process=self.process(b'{"complete":true')
        self.audit.process=process;self.audit.selector=selectors.DefaultSelector()
        self.audit.selector.register(process.stdout,selectors.EVENT_READ,'stdout')
        try:
            with self.assertRaisesRegex(RuntimeError,'EOF'):self.audit.read_frame(.2)
        finally:
            self.audit.selector.close();process.stdout.close();process.stderr.close()
        self.audit.pending=client.reader.encode_frame(self.frame);self.audit.process=SimpleNamespace(wait=lambda timeout:2)
        with self.assertRaisesRegex(RuntimeError,'unsuccessfully'):self.audit.snapshot()

    def test_frame_cap_and_wrong_request_scope_rejected_without_truncating(self):
        self.audit.pending=b'x'*(client.reader.MAX_FRAME+1)
        with self.assertRaisesRegex(RuntimeError,'cap'):self.audit.read_frame(.2)
        self.audit.pending=client.reader.encode_frame(dict(self.frame,complete=False))
        with self.assertRaisesRegex(RuntimeError,'snapshot'):self.audit.snapshot()

    def test_cleanup_failure_is_recorded_and_idempotent_never_masks_active(self):
        self.audit.run.mkdir();self.audit.container=self.identifier;self.audit.validated=True
        with patch.object(client.subprocess,'run',side_effect=TimeoutError('synthetic stop')) as command:
            with self.assertRaises(RuntimeError):self.audit.close()
            self.audit.close()
        self.assertEqual(command.call_count,1)
        self.assertFalse(json.loads((self.audit.run/'receipt.json').read_text())['removed'])

    def test_fresh_external_scope_counts_whole_nv_parent_and_published_tmp_with_dedup(self):
        parent=self.root/'production-cache';vllm=parent/'vllm';vllm.mkdir(parents=True)
        nv=self.root/'nv';nv.mkdir();(nv/'kernel').write_bytes(b'owned-kernel')
        other=parent/'numba';other.mkdir();os.link(nv/'kernel',other/'same-inode')
        temporary=self.root/'owned-tmp';temporary.write_bytes(b'temporary')
        publication=self.root/'publication.json'
        publication.write_text(json.dumps(dict(production_cache_fee_bytes=client.CHARGE_FLOOR,
            other_external_whole_roots=dict(allocated_bytes=3796992,roots={str(temporary):4096}))))
        ledger=self.root/'ledger.json';ledger.write_text(json.dumps(dict(schema='aegis.external-cache-ledger.v1',
            goal_start_epoch=1793686024,temporary_scope_complete=True,owned_temporary_paths=[str(temporary)],
            minimum_additional_external_bytes=3796992,source_receipt=str(publication),source_receipt_sha256=client.sha(publication))))
        with patch.object(client.reader,'HOST_CACHE',str(vllm)),patch.object(client,'NV_ROOT',nv),patch.object(client,'PUBLICATION_PATH',publication):
            result=client.extra_external(ledger,[])
            expected=sum(p.stat().st_blocks*512 for p in (nv,nv/'kernel',other,temporary))
            self.assertEqual(result['measured_full_bytes'],expected)
            self.assertEqual(result['accounted_bytes'],3796992)
            value=json.loads(ledger.read_text());value['owned_temporary_paths']=[];ledger.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'omitted'):client.extra_external(ledger,[])

    def test_guard_failure_before_create_cannot_launch_docker(self):
        audit=client.Auditor(self.root/'denied',lambda:(_ for _ in ()).throw(RuntimeError('not owned')))
        with patch.object(client.subprocess,'run') as command,patch.object(client.subprocess,'Popen') as start:
            with self.assertRaisesRegex(RuntimeError,'not owned'):audit.__enter__()
        command.assert_not_called();start.assert_not_called()


if __name__=='__main__':unittest.main()
