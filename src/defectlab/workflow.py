"""Read-only next actions for a versioned plan and evidence-bound research ledger."""
from pathlib import Path

from . import governance
from .common import hash_file, load_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plan(path):
    plan = load_json(path)
    _require(isinstance(plan, dict) and plan.get('schema_version') == 1,
             'unsupported workflow plan schema')
    _require(isinstance(plan.get('workflow_id'), str) and plan['workflow_id'].strip(),
             'workflow_id is required')
    steps = plan.get('steps')
    _require(isinstance(steps, list) and steps, 'workflow steps must be a nonempty list')
    index = {}
    for step in steps:
        _require(isinstance(step, dict), 'workflow step must be an object')
        for key in ('id', 'title', 'domain', 'next_action'):
            _require(isinstance(step.get(key), str) and step[key].strip(), f'step requires {key}')
        _require(step['id'] not in index, 'duplicate workflow step ID')
        for key in ('depends_on', 'completion_conditions'):
            values = step.get(key)
            _require(isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values),
                     f'step {key} must be a string list')
            _require(len(values) == len(set(values)), f'duplicate step {key}')
        _require(step['completion_conditions'], 'step requires completion conditions')
        for key in ('requires_review', 'optional'):
            _require(key not in step or type(step[key]) is bool, f'step {key} must be boolean')
        index[step['id']] = step
    visited, active, ordered = set(), set(), []

    def visit(sid):
        _require(sid in index, f'unknown workflow dependency: {sid}')
        _require(sid not in active, 'cyclic workflow dependencies')
        if sid in visited:
            return
        active.add(sid)
        for dependency in index[sid]['depends_on']:
            visit(dependency)
        active.remove(sid)
        visited.add(sid)
        ordered.append(index[sid])

    for sid in index:
        visit(sid)
    return plan, ordered


def _reachable(record, index):
    pending, visited = [record['id']], set()
    while pending:
        rid = pending.pop()
        if rid in visited:
            continue
        visited.add(rid)
        item = index[rid]
        pending.extend(ref['id'] for field in ('refs', 'depends_on') for ref in item.get(field, []))
        # Execution/request provenance is relevant even if not duplicated in refs.
        pending.extend(item[field] for field in ('task_id', 'request_id', 'execution_id') if item.get(field))
    return visited


def _completed_execution(item):
    return (item.get('kind') == 'execution' and item.get('status') in ('completed', 'responded')
            and bool(item.get('host_task_handle')) and bool(item.get('evidence')))


def _review_valid(review, index, reachable):
    execution = index.get(review.get('execution_id'), {})
    targets = [index[ref['id']] for ref in review.get('refs', [])]
    return (review.get('kind') == 'review' and review.get('independent') is True
            and bool(review.get('author')) and bool(review.get('reviewer'))
            and review['author'] != review['reviewer'] and bool(targets)
            and _completed_execution(execution) and execution['id'] in reachable
            and all(target.get('owner') != review['reviewer']
                    and target.get('author') != review['reviewer']
                    and target.get('execution_id') != execution['id'] for target in targets))


def resume(plan_path, ledger_path):
    """Describe ready work without invoking experts, changing state or doing research.

    Completion is an explicit, current, fresh stage record for this exact plan file.
    It pins each latest completed prerequisite, an execution and a decision. Reviews
    are additionally required when the plan declares them. Missing history is unknown.
    """
    plan_path, ledger_path = Path(plan_path), Path(ledger_path)
    plan_hash = hash_file(plan_path)
    available = ledger_path.is_file()
    ledger_hash = hash_file(ledger_path) if available else None
    plan, steps = _plan(plan_path)
    ledger = governance._ledger(ledger_path) if available else {'records': []}
    records = ledger['records']
    index = {}
    for item in records:
        governance._references(item, index)
        index[item['id']] = item
    history_available = available and bool(records)
    assessment = governance.status(ledger_path) if history_available else None
    latest = {item['stage_id']: item for item in records if item['kind'] == 'stage'}
    evaluated, rows = {}, []
    stale = assessment['evidence_freshness']['stale_records'] if assessment else {}
    changed = assessment['needs_reassessment'] if assessment else {}
    for step in steps:
        sid = step['id']
        stage = latest.get(sid)
        dependencies = [evaluated[dep] for dep in step['depends_on']]
        blocked_by = [dep['id'] for dep in dependencies if dep['readiness'] != 'done']
        reasons = []
        validity = 'unknown'
        done = False
        reachable = _reachable(stage, index) if stage else set()
        if stage:
            validity = stage.get('validity', 'unknown')
            if validity != 'current':
                reasons.append('stage validity is not explicitly current')
            if stage.get('workflow_plan_sha256') != plan_hash:
                reasons.append('workflow plan version is not pinned to the current file hash')
            if reachable & set(stale):
                reasons.append('reachable evidence is missing or changed')
            if reachable & set(changed):
                reasons.append('reachable records require reassessment')
            for dep in step['depends_on']:
                parent = latest.get(dep)
                if parent is None or parent['id'] not in reachable:
                    reasons.append(f'latest prerequisite version is not pinned: {dep}')
            if stage.get('status') == 'completed':
                if not set(step['completion_conditions']).issubset(stage.get('completion_conditions', [])):
                    reasons.append('recorded completion conditions do not cover the current plan')
                reachable_items = [index[rid] for rid in reachable]
                stage_evidence = {entry['sha256'] for entry in stage.get('evidence', [])}
                executions = [item for item in reachable_items if _completed_execution(item)
                              and stage_evidence & {entry['sha256'] for entry in item['evidence']}]
                decisions = [item for item in reachable_items if item.get('kind') == 'decision']
                qualifying_decisions = []
                for decision in decisions:
                    decision_ids = _reachable(decision, index)
                    covered = {entry['sha256'] for item in executions if item['id'] in decision_ids
                               for entry in item['evidence']}
                    if stage_evidence and stage_evidence.issubset(covered):
                        qualifying_decisions.append(decision)
                if not qualifying_decisions:
                    reasons.append('completion requires executions and a decision covering all stage evidence')
                if not stage.get('evidence') or not stage.get('completion_conditions'):
                    reasons.append('stage completion lacks conditions or evidence')
                if step.get('requires_review'):
                    reviews = [item for item in reachable_items if _review_valid(item, index, reachable)]
                    # The review must be used by a decision and cover this stage's evidence.
                    stage_evidence = {entry['sha256'] for entry in stage.get('evidence', [])}
                    reviewed = False
                    for review in reviews:
                        target_ids = {ref['id'] for ref in review['refs']}
                        target_evidence = {entry['sha256'] for rid in target_ids
                                           for target_id in _reachable(index[rid], index)
                                           for entry in index[target_id].get('evidence', [])}
                        if stage_evidence.issubset(target_evidence) and any(
                                review['id'] in _reachable(decision, index)
                                for decision in qualifying_decisions):
                            reviewed = True
                    if not reviewed:
                        reasons.append('completion requires a decision using an independent review of its evidence')
                done = not reasons and not blocked_by
            if reasons:
                validity = ('unknown' if stage.get('validity', 'unknown') == 'unknown'
                            and len(reasons) == 1 else 'needs_reassessment')
        if not history_available:
            reasons = ['research history unavailable; locate handoff or explicitly record a new session']
            readiness = 'blocked'
        elif blocked_by:
            readiness = 'blocked'
            reasons.append('prerequisites are not currently complete: ' + ', '.join(blocked_by))
        elif stage and stage.get('status') in ('blocked', 'awaiting_input') and not reasons:
            readiness = 'blocked'
            reasons.append(stage.get('reason', 'recorded stage awaits input'))
        elif done:
            readiness = 'done'
        elif stage and reasons:
            readiness = 'needs_reassessment'
        else:
            readiness = 'ready'
        row = {'id': sid, 'title': step['title'], 'domain': step['domain'],
               'optional': step.get('optional', False), 'requires_review': step.get('requires_review', False),
               'record_id': stage['id'] if stage else None,
               'historical_status': stage.get('status') if stage else None,
               'validity': validity, 'readiness': readiness, 'reasons': reasons,
               'blocked_by': blocked_by, 'next_action': step['next_action'],
               'completion_conditions': step['completion_conditions']}
        rows.append(row)
        evaluated[sid] = row
    actionable = [row for row in rows if row['readiness'] in ('ready', 'needs_reassessment')]
    prioritized = sorted(actionable, key=lambda row: row['optional'])
    next_step = prioritized[0] if prioritized else None
    result = {'schema_version': 1, 'workflow_id': plan['workflow_id'], 'workflow_plan_sha256': plan_hash,
            'ledger_sha256': ledger_hash,
            'history_available': history_available,
            'history_status': 'available' if history_available else 'unknown',
            'preparatory_action': None if history_available else (
                'Locate the private handoff/ledger first. If unavailable, explicitly record a new session '
                'with unknown prior research history using defectlab record; no history is reconstructed.'),
            'configured_model': plan.get('model'), 'configured_reasoning_effort': plan.get('reasoning_effort'),
            'observed_model': None, 'observed_reasoning_effort': None,
            'steps': rows, 'ready_steps': [row['id'] for row in actionable], 'next_step': next_step,
            'complete': history_available and all(row['readiness'] == 'done' for row in rows if not row['optional']),
            'limitations': ['Read-only declared dependencies; no expert, browser, upload or execution is invoked.',
                            'Current evidence and recorded independence do not prove scientific validity.',
                            'Optional stages remain explicit; skipping them does not manufacture completion.',
                            'Actual model/effort and remote persistence are not inferred.']}
    # Ledger writers replace atomically; separate reads must not mix generations.
    # No lock file is opened, so a missing-history read remains strictly read-only.
    _require(plan_path.is_file() and hash_file(plan_path) == plan_hash,
             'workflow plan changed during assessment; retry')
    _require(ledger_path.is_file() == available
             and (hash_file(ledger_path) if available else None) == ledger_hash,
             'workflow ledger changed during assessment; retry')
    return result
