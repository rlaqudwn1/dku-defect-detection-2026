"""Resumption must preserve unknown history and invalidate stale completion."""
import json
from pathlib import Path

import pytest

from defectlab import governance
from defectlab.cli import parser, dispatch
from defectlab.common import hash_file
from defectlab.workflow import resume


@pytest.fixture
def work(tmp_path):
    plan = tmp_path / 'plan.json'
    steps = [dict(id='A', title='Input check', domain='data', depends_on=[],
                  completion_conditions=['inputs fixed'], next_action='Verify inputs'),
             dict(id='B', title='Decision', domain='evaluation', depends_on=['A'],
                  completion_conditions=['decision checked'], next_action='Review decision',
                  requires_review=True),
             dict(id='C', title='Visual optional', domain='data', depends_on=['A'],
                  completion_conditions=['visual'], next_action='Inspect permitted sample', optional=True)]
    plan.write_text(json.dumps(dict(schema_version=1, workflow_id='test', steps=steps)))
    ledger = tmp_path / 'ledger.json'
    counter = 0

    def add(**item):
        nonlocal counter
        counter += 1
        path = tmp_path / f'input-{counter}.json'
        path.write_text(json.dumps(item))
        return governance.record(ledger, path)

    def evidence(name, body='result'):
        path = tmp_path / name
        path.write_text(body)
        return {'path': str(path), 'sha256': hash_file(path)}

    def pin(item):
        return {'id': item['id'], 'sha256': item['sha256']}

    def complete(sid, deps=(), review=False, suffix='', validity='current'):
        spec = next(step for step in steps if step['id'] == sid)
        stem = sid + suffix
        proof = evidence(stem + '-result.txt', stem)
        task = add(id=stem + '-task', kind='task', owner='analyst', status='in_progress', evidence=[proof])
        add(id=stem + '-request', kind='request', task_id=task['id'])
        execution = add(id=stem + '-execution', kind='execution', request_id=stem + '-request',
                        status='completed', host_task_handle='actual-local-call', evidence=[proof])
        refs = [pin(task), pin(execution)] + [pin(dep) for dep in deps]
        if review:
            add(id=stem + '-review-request', kind='request', task_id=task['id'])
            rexe = add(id=stem + '-review-execution', kind='execution', request_id=stem + '-review-request',
                       status='completed', host_task_handle='separate-review-call',
                       evidence=[evidence(stem + '-review.txt', 'review')])
            peer = add(id=stem + '-review', kind='review', execution_id=rexe['id'], independent=True,
                       author='analyst', reviewer='peer', refs=[pin(task)], evidence=rexe['evidence'])
            refs.append(pin(peer))
        decision = add(id=stem + '-decision', kind='decision', owner='conductor', refs=refs)
        return add(id=stem + '-stage', kind='stage', stage_id=sid, owner='conductor', reason='observed',
                   status='completed', validity=validity, workflow_plan_sha256=hash_file(plan),
                   completion_conditions=spec['completion_conditions'], evidence=[proof], refs=[pin(decision)])

    return plan, ledger, steps, add, evidence, pin, complete


def test_missing_and_empty_history_are_unknown_read_only(work):
    plan, ledger, _, _, _, _, _ = work
    result = resume(plan, ledger)
    assert not result['history_available']
    assert result['next_step'] is None
    assert result['ledger_sha256'] is None
    assert all(row['validity'] == 'unknown' and row['readiness'] == 'blocked' for row in result['steps'])
    assert not ledger.exists()
    ledger.write_text('{"schema_version": 1, "records": []}')
    assert resume(plan, ledger)['history_status'] == 'unknown'


def test_happy_resume_optional_completion_and_cli(work):
    plan, ledger, _, add, _, _, complete = work
    add(id='session', kind='session', status='started', prior_history='unknown')
    assert resume(plan, ledger)['next_step']['id'] == 'A'
    a = complete('A')
    result = dispatch(parser().parse_args(['workflow', str(plan), str(ledger)]))
    assert result['ready_steps'] == ['B', 'C']
    assert result['next_step']['id'] == 'B'
    complete('B', [a], review=True)
    result = resume(plan, ledger)
    assert result['complete']
    assert result['steps'][2]['readiness'] == 'ready'
    assert result['observed_model'] is None


def test_stale_dependency_blocks_descendant_but_unrelated_old_evidence_does_not(work):
    plan, ledger, _, add, evidence, _, complete = work
    old = evidence('old.txt')
    add(id='old', kind='task', evidence=[old])
    a = complete('A')
    complete('B', [a], review=True)
    Path(old['path']).unlink()
    assert resume(plan, ledger)['complete']
    Path(a['evidence'][0]['path']).write_text('changed')
    result = resume(plan, ledger)
    assert result['steps'][0]['readiness'] == 'needs_reassessment'
    assert result['steps'][1]['readiness'] == 'blocked'
    assert result['steps'][1]['historical_status'] == 'completed'


def test_updated_prerequisite_and_plan_versions_do_not_inherit_completion(work):
    plan, ledger, _, _, _, _, complete = work
    a = complete('A')
    complete('B', [a], review=True)
    complete('A', suffix='-new')
    result = resume(plan, ledger)
    assert result['steps'][1]['readiness'] == 'needs_reassessment'
    assert any('latest prerequisite' in reason for reason in result['steps'][1]['reasons'])
    obj = json.loads(plan.read_text())
    obj['steps'][0]['next_action'] = 'Changed action'
    plan.write_text(json.dumps(obj))
    assert resume(plan, ledger)['steps'][0]['readiness'] == 'needs_reassessment'


def test_unknown_validity_and_missing_review_fail_closed(work):
    plan, ledger, _, _, _, _, complete = work
    a = complete('A', validity='unknown')
    assert resume(plan, ledger)['steps'][0]['readiness'] == 'needs_reassessment'
    a = complete('A', suffix='-known')
    complete('B', [a])
    result = resume(plan, ledger)
    assert result['steps'][1]['readiness'] == 'needs_reassessment'
    assert any('independent review' in reason for reason in result['steps'][1]['reasons'])


def test_blocked_browser_does_not_stop_parallel_local_branch(work):
    plan, ledger, steps, add, _, _, complete = work
    steps[1]['depends_on'] = []
    steps[2]['optional'] = False
    plan.write_text(json.dumps(dict(schema_version=1, workflow_id='test', steps=steps)))
    complete('A')
    add(id='blocked-browser', kind='stage', stage_id='B', status='blocked', owner='conductor',
        reason='login requires user', validity='current', workflow_plan_sha256=hash_file(plan))
    result = resume(plan, ledger)
    assert result['steps'][1]['readiness'] == 'blocked'
    assert result['next_step']['id'] == 'C'


@pytest.mark.parametrize('mutation', ['duplicate', 'unknown', 'cycle'])
def test_invalid_plan_graph_is_rejected(work, mutation):
    plan, ledger, steps, *_ = work
    if mutation == 'duplicate':
        steps.append(dict(steps[0]))
    elif mutation == 'unknown':
        steps[0]['depends_on'] = ['missing']
    else:
        steps[0]['depends_on'] = ['B']
    plan.write_text(json.dumps(dict(schema_version=1, workflow_id='test', steps=steps)))
    with pytest.raises(ValueError):
        resume(plan, ledger)


def test_ancestor_review_and_execution_cannot_complete_unrelated_work(work):
    plan, ledger, steps, add, evidence, pin, complete = work
    a = complete('A', review=True)
    # A legitimate older review in the dependency closure does not review B's result.
    complete('B', [a])
    result = resume(plan, ledger)
    assert result['steps'][1]['readiness'] == 'needs_reassessment'
    assert any('review of its evidence' in reason for reason in result['steps'][1]['reasons'])
    proof = evidence('unexecuted-c.txt', 'claimed outcome with no invocation')
    add(id='unexecuted-c', kind='stage', stage_id='C', owner='conductor', reason='claim',
        status='completed', validity='current', workflow_plan_sha256=hash_file(plan),
        evidence=[proof], completion_conditions=steps[2]['completion_conditions'], refs=[pin(a)])
    result = resume(plan, ledger)
    assert result['steps'][2]['readiness'] == 'needs_reassessment'
    assert any('executions and a decision' in reason for reason in result['steps'][2]['reasons'])


def test_shared_reviewed_input_does_not_cover_new_unreviewed_output(work):
    plan, ledger, steps, add, evidence, pin, complete = work
    a = complete('A')
    shared = evidence('shared.txt', 'reviewed input')
    output = evidence('new-output.txt', 'unreviewed derived decision')
    task = add(id='partial-task', kind='task', owner='analyst', evidence=[shared])
    add(id='partial-request', kind='request', task_id=task['id'])
    execution = add(id='partial-execution', kind='execution', request_id='partial-request',
                    status='completed', host_task_handle='actual', evidence=[shared, output])
    add(id='partial-review-request', kind='request', task_id=task['id'])
    rexe = add(id='partial-review-execution', kind='execution', request_id='partial-review-request',
               status='completed', host_task_handle='review-call', evidence=[evidence('partial-review.txt')])
    review = add(id='partial-review', kind='review', execution_id=rexe['id'], independent=True,
                 author='analyst', reviewer='peer', refs=[pin(task)])
    decision = add(id='partial-decision', kind='decision', refs=[pin(execution), pin(review), pin(a)])
    add(id='partial-stage', kind='stage', stage_id='B', owner='conductor', reason='claimed',
        status='completed', validity='current', workflow_plan_sha256=hash_file(plan),
        completion_conditions=steps[1]['completion_conditions'], evidence=[shared, output], refs=[pin(decision)])
    result = resume(plan, ledger)
    assert result['steps'][1]['readiness'] == 'needs_reassessment'
    assert any('review of its evidence' in reason for reason in result['steps'][1]['reasons'])
    # Nor does one executed input cover an extra stage output without execution evidence.
    borrowed = add(id='only-input-decision', kind='decision', refs=[pin(a)])
    add(id='only-input-stage', kind='stage', stage_id='C', owner='conductor', reason='claimed',
        status='completed', validity='current', workflow_plan_sha256=hash_file(plan),
        completion_conditions=steps[2]['completion_conditions'],
        evidence=a['evidence'] + [output], refs=[pin(borrowed)])
    result = resume(plan, ledger)
    assert result['steps'][2]['readiness'] == 'needs_reassessment'
    assert any('all stage evidence' in reason for reason in result['steps'][2]['reasons'])


def test_concurrent_ledger_append_fails_closed_instead_of_mixing_snapshots(work, monkeypatch):
    plan, ledger, _, _, _, _, complete = work
    a = complete('A')
    complete('B', [a], review=True)
    original_status = governance.status

    def append_during_assessment(path):
        assessment = original_status(path)
        complete('A', suffix='-concurrent')
        return assessment

    monkeypatch.setattr(governance, 'status', append_during_assessment)
    with pytest.raises(ValueError, match='ledger changed during assessment; retry'):
        resume(plan, ledger)


def test_concurrent_plan_change_fails_closed(work, monkeypatch):
    plan, ledger, _, _, _, _, complete = work
    complete('A')
    original_status = governance.status

    def change_plan_during_assessment(path):
        assessment = original_status(path)
        plan.write_text(plan.read_text() + '\n')
        return assessment

    monkeypatch.setattr(governance, 'status', change_plan_during_assessment)
    with pytest.raises(ValueError, match='plan changed during assessment; retry'):
        resume(plan, ledger)


def test_concurrent_missing_ledger_creation_fails_closed(work, monkeypatch):
    from defectlab import workflow
    plan, ledger, _, add, *_ = work
    original_plan = workflow._plan

    def create_ledger_during_assessment(path):
        result = original_plan(path)
        add(id='new-session', kind='session', status='started', prior_history='unknown')
        return result

    monkeypatch.setattr(workflow, '_plan', create_ledger_during_assessment)
    with pytest.raises(ValueError, match='ledger changed during assessment; retry'):
        resume(plan, ledger)
