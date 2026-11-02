"""Scientific-report guards: don't rank unmatched or unfinished experiments."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('audit_summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_wall_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_rank_correlation_handles_ties_and_undefined_constant_targets():
    assert audit.spearman([1, 1, 3], [7, 7, 9]) == pytest.approx(1)
    assert audit.spearman([1, 2, 3], [9, 8, 7]) == pytest.approx(-1)
    assert audit.spearman([1, 2, 3], [0, 0, 0]) is None


def test_perfect_small_sample_is_not_certain():
    low, high = audit.wilson([True]*32)
    assert 0.89 < low < 0.9
    assert high == pytest.approx(1)


def test_unmatched_counterfactuals_cannot_influence_timing_claims(tmp_path):
    rows = [dict(episode=0,t=t,prefix_ok=True,prediction=t,real_q_gap=t,return_drop=t) for t in [1,2,3]]
    rows += [dict(episode=1,t=t,prefix_ok=False,prediction=t,real_q_gap=-t,return_drop=-t) for t in [1,2,3]]
    task = dict(id='fixture',kind='branch',map='3m',output=str(tmp_path/'result.json'))
    task_path = tmp_path/'task.json'
    task_path.write_text(json.dumps(task))
    (tmp_path/'manifest.json').write_text(json.dumps({'tasks':[str(task_path)]}))
    result = dict(task=task,status='completed',episodes=[],branches=rows)
    (tmp_path/'result.json').write_text(json.dumps(result))
    s = audit.summarize(tmp_path)
    assert s['branches']['3m']['tested'] == 6
    assert s['branches']['3m']['matched'] == 3
    assert s['branches']['3m']['metrics']['rank_q']['mean'] == pytest.approx(1)
    result['status'] = 'running'
    (tmp_path/'result.json').write_text(json.dumps(result))
    assert audit.summarize(tmp_path)['completed_tasks'] == 0
