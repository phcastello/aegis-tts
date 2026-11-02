"""Outage accounting must retain recovery windows and the latest service state."""
import json
from pathlib import Path

from training.f5_report import collect, operation_windows


def write(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/name).write_text(json.dumps(value))


def window(root, name, start, end=None, restored_id=None):
    directory=root/name
    record=dict(started_utc=start,stop_started_utc=start,stopped_utc=start)
    if end:
        record.update(restored_utc=end,gpu_work_ended_utc=end,status='production_restored')
        write(directory,'restored.json',dict(container_id=restored_id,running=True,health='healthy'))
    write(directory,'gpu-window.json',record)
    write(directory,'baseline.json',dict(container_id=name))
    return directory


def test_two_completed_windows_sum_only_actual_outages_and_use_latest_restore(tmp_path):
    first=window(tmp_path,'operations','2026-11-02T10:00:00+00:00','2026-11-02T10:05:00+00:00','first')
    record=json.loads((first/'gpu-window.json').read_text())
    record['error']='recoverable evaluator import failure'
    write(first,'gpu-window.json',record)
    window(tmp_path,'operations-resume-01','2026-11-02T10:10:00+00:00','2026-11-02T10:20:00+00:00','final')
    result=operation_windows(tmp_path)
    assert result['downtime_seconds']==900  # Excludes the five-minute restored interval.
    assert result['gpu_experiment_seconds']==900
    assert result['restored']['container_id']=='final'
    assert result['baseline']['container_id']=='operations'
    assert result['production_restoration_verified']
    assert not result['downtime_in_progress']
    assert len(result['windows'])==2
    assert result['intermediate_restorations'][0]['preceding_experiment_error']=='recoverable evaluator import failure'


def test_active_resumed_window_does_not_reuse_earlier_success(tmp_path):
    window(tmp_path,'operations','2026-11-02T10:00:00+00:00','2026-11-02T10:05:00+00:00','first')
    window(tmp_path,'operations-resume-01','2026-11-02T10:10:00+00:00')
    report,_,_=collect(tmp_path)
    result=report['operation']
    assert result['downtime_seconds'] is None
    assert result['downtime_completed_windows_seconds']==300
    assert result['restored'] is None
    assert result['downtime_in_progress']
    assert not result['production_restoration_verified']
    assert 'production_restoration_verified' in report['pending']


def test_windows_order_by_actual_time_and_ignore_unused_snapshot(tmp_path):
    window(tmp_path,'operations','2026-11-02T10:00:00Z','2026-11-02T10:05:00Z','first')
    window(tmp_path,'operations-resume-10','2026-11-02T10:10:00Z','2026-11-02T10:20:00Z','final')
    window(tmp_path,'operations-resume-2','2026-11-02T10:06:00Z','2026-11-02T10:07:00Z','middle')
    write(tmp_path/'operations-resume-99','snapshot.json',dict(container_id='unused'))
    result=operation_windows(tmp_path)
    assert result['restored']['container_id']=='final'
    assert result['downtime_seconds']==960
    assert len(result['windows'])==3


def test_acoustics_manifest_cannot_satisfy_listening_package_completion(tmp_path):
    write(tmp_path/'acoustics','manifest-sha256.json',{'f0.png':'example'})
    write(tmp_path/'acoustics','results.json',{'owner_perceptual_approval':False})
    report,_,_=collect(tmp_path)
    assert 'listening_package' in report['pending']
    assert report['listening_manifests']==[]
    assert report['acoustics']=={'owner_perceptual_approval':False}
