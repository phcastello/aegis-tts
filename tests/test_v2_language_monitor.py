import tempfile
import unittest
from pathlib import Path

from training.v2_language_monitor import Monitor, job_key, physical_usage, retained_jobs, read


class LocalMonitorTests(unittest.TestCase):
    def test_new_audio_or_scoring_policy_gets_new_key(self):
        base=job_key('manifest',{'case':'audio'},'corpus','spoken','scorer')
        self.assertEqual(base,job_key('manifest',{'case':'audio'},'corpus','spoken','scorer'))
        self.assertNotEqual(base,job_key('manifest',{'case':'changed'},'corpus','spoken','scorer'))
        self.assertNotEqual(base,job_key('manifest',{'case':'audio'},'corpus','spoken','new-scorer'))

    def test_retention_protects_initial_critical_and_active_older_epochs(self):
        jobs=[dict(key=str(i),label=f'epoch-{i:04}',epoch=i,step=i*3000) for i in range(12)]
        jobs.append(dict(key='initial',label='initial',epoch=0,step=0))
        self.assertEqual(retained_jobs(jobs,{'1'},'2'),{'initial','1','2',*map(str,range(6,12))})

    def test_hardlinked_shared_audio_is_counted_once_across_roots(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            wav=a/'wave';wav.write_bytes(b'x'*9000);os.link(wav,b/'snapshot')
            usage=physical_usage([a,b])
            expected=sum(p.stat().st_blocks*512 for p in (a,b,wav))
            self.assertEqual(usage['allocated_bytes'],expected)

    def test_new_epoch_review_survives_notification_logging(self):
        import logging
        with tempfile.TemporaryDirectory() as temp:
            monitor=Monitor.__new__(Monitor);monitor.run=Path(temp)
            monitor.log=logging.getLogger('monitor-test')
            monitor.review('new_epoch','epoch-hash',epoch=1,step=3000)
            notice=read(Path(temp)/'review-needed.json')['notices'][0]
            self.assertEqual((notice['kind'],notice['step']),('new_epoch',3000))


if __name__=='__main__':unittest.main()
