"""Meaningful artifact/denominator/budget contracts, stdlib and tiny synthetic WAVs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training import v2_fastpitch_common_cpu as control
from training import v2_language_monitor as ledger


class CommonCpuTests(unittest.TestCase):
    def artifact(self,root,step=10000):
        source=root/'source.pt';source.write_bytes(b'tiny-synthetic-checkpoint')
        sha=control.common.sha(source);control.save(source.with_suffix('.sha256.json'),dict(step=step,sha256=sha,bytes=source.stat().st_size))
        output=root/'output';output.mkdir()
        return SimpleNamespace(checkpoint=source,checkpoint_sha256=sha,expected_step=10000,output=output)

    def test_freeze_binds_real_inode_and_rejects_wrong_step_before_link(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.artifact(Path(temp),step=9999)
            with self.assertRaisesRegex(ValueError,'step/SHA'):control.freeze(a)
            self.assertFalse((a.output/'checkpoint/checkpoint.pt').exists())
            control.save(a.checkpoint.with_suffix('.sha256.json'),dict(step=10000,sha256=a.checkpoint_sha256,bytes=a.checkpoint.stat().st_size))
            bound=control.freeze(a);frozen=Path(bound['path'])
            self.assertEqual(frozen.stat().st_ino,a.checkpoint.stat().st_ino);self.assertEqual(frozen.stat().st_nlink,2)
            expected=sum(p.stat().st_blocks*512 for p in a.output.rglob('*') if p.is_file() and p!=frozen)
            self.assertEqual(control.output_usage(a.output),expected)

    def test_cleanup_only_own_hardlink_preserves_source_and_defers_if_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.artifact(Path(temp));plan=dict(checkpoint=control.freeze(a));frozen=Path(plan['checkpoint']['path'])
            with patch.object(control.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=b'123',stderr=b'')):
                self.assertEqual(control.cleanup(a,plan)['status'],'preserved_for_review');self.assertTrue(frozen.exists())
            for stdout,stderr in ((b'123',b''),(b'',b'fuser failed')):
                with patch.object(control.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout=stdout,stderr=stderr)):
                    self.assertEqual(control.cleanup(a,plan)['status'],'preserved_for_review');self.assertTrue(frozen.exists())
            with patch.object(control.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout=b'',stderr=b'')):
                self.assertEqual(control.cleanup(a,plan)['status'],'removed_only_owned_temporary_hardlink')
            self.assertFalse(frozen.exists());self.assertEqual(a.checkpoint.read_bytes(),b'tiny-synthetic-checkpoint');self.assertEqual(a.checkpoint.stat().st_nlink,1)

    def test_cleanup_refuses_retainer_symlink_or_other_uid(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.artifact(Path(temp));plan=dict(checkpoint=control.freeze(a));frozen=Path(plan['checkpoint']['path'])
            original_lstat=Path.lstat
            def wrong_uid(path):
                st=original_lstat(path)
                return SimpleNamespace(st_mode=st.st_mode,st_uid=st.st_uid+1,st_dev=st.st_dev,st_ino=st.st_ino) if path==a.checkpoint else st
            with patch.object(Path,'lstat',wrong_uid),patch.object(control.subprocess,'run') as fuser:
                self.assertEqual(control.cleanup(a,plan)['status'],'preserved_for_review');fuser.assert_not_called()
            original=a.checkpoint.with_name('retained-original.pt');a.checkpoint.rename(original);a.checkpoint.symlink_to(original)
            with patch.object(control.subprocess,'run') as fuser:
                self.assertEqual(control.cleanup(a,plan)['status'],'preserved_for_review');fuser.assert_not_called()
            self.assertTrue(frozen.exists());self.assertEqual(original.stat().st_ino,frozen.stat().st_ino)

    def synthetic_rows(self,root):
        cases=[dict(id=str(i)) for i in range(7)];rows=[]
        for c in cases:
            path=root/(c['id']+'.wav');control.common.write_pcm(path,b'\x01\x00'*32)
            rows.append(dict(case=c['id'],outcome='ok',wav=str(path),wav_sha256=control.common.sha(path)))
        return rows,cases

    def test_all_seven_valid_pcm_conditions_required_not_filtered_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            rows,cases=self.synthetic_rows(Path(temp));control.validate_rows(rows,cases)
            for invalid in (rows[:-1],[dict(r,outcome='failed') if i==6 else r for i,r in enumerate(rows)],rows[:-1]+[rows[0]]):
                with self.assertRaisesRegex(ValueError,'All7'):control.validate_rows(invalid,cases)
            with __import__('wave').open(rows[0]['wav'],'wb') as wav:wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0'*32)
            rows[0]['wav_sha256']=control.common.sha(rows[0]['wav'])
            with self.assertRaisesRegex(ValueError,'PCM'):control.validate_rows(rows,cases)
            control.common.write_pcm(rows[0]['wav'],b'\x01\x00'*32)
            path=Path(rows[0]['wav']);path.write_bytes(path.read_bytes()[:-2]);rows[0]['wav_sha256']=control.common.sha(path)
            with self.assertRaisesRegex(ValueError,'Truncated'):control.validate_rows(rows,cases)

    def test_resource_abort_escapes_per_case_exception_but_runs_finally(self):
        events=[]
        try:
            try:
                try:control.interrupt()
                except Exception:events.append('incorrectly_swallowed')
                events.append('incorrectly_continued')
            finally:events.append('receipt_finally')
        except control.ControlAbort:events.append('aborted')
        self.assertEqual(events,['receipt_finally','aborted'])

    def asr_fixture(self):
        jobs=[dict(candidate='fixed',case=str(i),input_audio_sha256=str(i),expected_verbatim='texto',expected_spoken='texto',expected_effective='texto') for i in range(7)]
        plan=dict(asr=dict(receipt_sha256='receipt',settings=dict(device='cpu',cpu_threads=2),versions=dict(recognizer='pinned')))
        attempts=[dict(**j,audio_sha256=j['input_audio_sha256'],success=True,comparisons=dict(verbatim=dict(word_edits=1,reference_tokens=['texto'],character_edits=1,reference_characters=5))) for j in jobs]
        receipt=dict(script_sha256=control.SCORER_SHA,model_receipt_sha256='receipt',completed_utc='done',settings=plan['asr']['settings'],versions=plan['asr']['versions'],attempts=attempts)
        return receipt,jobs,plan

    def test_asr_requires_all_seven_successes_exact_text_audio_and_versions(self):
        receipt,jobs,plan=self.asr_fixture();result=control.audit_asr(receipt,jobs,plan);self.assertEqual((result['word_edits'],result['reference_words']),(7,7))
        for mutation in ('missing','failed','duplicate','text','version','scorer'):
            invalid=copy.deepcopy(receipt)
            if mutation=='missing':invalid['attempts'].pop()
            elif mutation=='failed':invalid['attempts'][0]['success']=False
            elif mutation=='duplicate':invalid['attempts'][-1]=invalid['attempts'][0]
            elif mutation=='text':invalid['attempts'][0]['expected_verbatim']='alterado'
            elif mutation=='version':invalid['versions']['recognizer']='other'
            else:invalid['script_sha256']='other'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):control.audit_asr(invalid,jobs,plan)

    def test_preflight_includes_git_and_exact_cap_reserves(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=SimpleNamespace(output=root);usage=dict(allocated_bytes=39_950_000_000,permission_errors=[]);git=root/'git-common'
            with patch.object(ledger,'physical_usage',return_value=usage) as measure,patch.object(control,'git_common',return_value=git),patch.object(control.shutil,'disk_usage',return_value=SimpleNamespace(free=5*1024**3+control.DISK_CAP)):
                result=control.preflight(a);self.assertIn(git,measure.call_args.args[0]);self.assertEqual(result['extra_output_bytes'],50_000_000)
                usage['allocated_bytes']+=1
                with self.assertRaisesRegex(RuntimeError,'Operational'):control.preflight(a)

    def test_final_flag_rejected_before_files_or_heavy_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'must-not-exist'
            command=[sys.executable,'-m','training.v2_fastpitch_common_cpu','--plan','--expected-step','10000','--checkpoint','/nonexistent','--checkpoint-sha256','a'*64,'--output',str(output),'--final']
            result=subprocess.run(command,cwd=control.REPO,capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertIn('unrecognized arguments: --final',result.stderr);self.assertFalse(output.exists())
            result=subprocess.run([sys.executable,'-c',"import sys; from training import v2_fastpitch_common_cpu; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],cwd=control.REPO,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
