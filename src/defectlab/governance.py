"""Local, evidence-bound research records. No agent or remote service is invoked."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .common import hash_file, hash_json, load_json

KINDS = {'task', 'request', 'execution', 'review', 'decision', 'stage', 'change',
         'time', 'session', 'reservation', 'receipt'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ledger(path):
    if not Path(path).exists():
        return {'schema_version': 1, 'records': []}
    ledger = load_json(path)
    _require(ledger.get('schema_version') == 1 and isinstance(ledger.get('records'), list),
             'unsupported ledger schema')
    seen = set()
    for item in ledger['records']:
        _require(item.get('id') not in seen, 'duplicate ledger ID')
        seen.add(item['id'])
        _require(item.get('sha256') == hash_json({k: v for k, v in item.items() if k != 'sha256'}),
                 'ledger record hash mismatch')
    return ledger


def _stamp(value):
    _require(isinstance(value, str), 'timestamp must be an ISO string with timezone')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _require(parsed.tzinfo is not None, 'timestamp requires timezone')
    return parsed.timestamp()


def _number(value, label):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value >= 0, f'{label} must be finite and nonnegative')


def _evidence(items, base, verify=True):
    _require(isinstance(items, list), 'evidence must be a list')
    result = []
    for item in items:
        _require(isinstance(item, dict) and item.get('path') and item.get('sha256'),
                 'evidence requires path and sha256')
        entry = dict(item)
        path = Path(entry['path'])
        if not path.is_absolute():
            path = base / path
        entry['path'] = str(path.resolve())
        if verify:
            _require(path.is_file() and hash_file(path) == entry['sha256'],
                     f'evidence missing or changed: {path}')
        result.append(entry)
    return result


def _references(item, index):
    targets = []
    for field in ('refs', 'depends_on'):
        _require(isinstance(item.get(field, []), list), f'{field} must be a list')
        for ref in item.get(field, []):
            _require(isinstance(ref, dict) and ref.get('id') in index, 'unknown record reference')
            _require(ref.get('sha256') == index[ref['id']]['sha256'], 'reference version hash mismatch')
            targets.append(ref['id'])
    for field in ('task_id', 'request_id', 'execution_id', 'supersedes'):
        if item.get(field):
            _require(item[field] in index, f'unknown {field}')
            targets.append(item[field])
    return targets


def record(ledger_path, record_path):
    """Validate and append one immutable JSON record under a process lock.

    IDs identify immutable versions. ``refs``/``depends_on`` pin record hashes;
    evidence paths are relative to the input JSON. Amendments get new IDs and
    ``supersedes``. Actual agent metadata is never inferred from requested data.
    """
    ledger_path, record_path = Path(ledger_path), Path(record_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(ledger_path) + '.lock', 'a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _ledger(ledger_path)
        item = dict(load_json(record_path))
        _require(isinstance(item.get('id'), str) and item['id'].strip(), 'record requires id')
        _require(item.get('kind') in KINDS, 'unknown record kind')
        _require('sha256' not in item, 'input record hash is computed, not supplied')
        index = {r['id']: r for r in ledger['records']}
        _require(item['id'] not in index, 'duplicate record ID')
        _references(item, index)
        item['evidence'] = _evidence(item.get('evidence', []), record_path.parent)
        kind = item['kind']
        if kind == 'request':
            _require(item.get('status', 'planned') not in ('reviewed', 'completed'),
                     'request preparation is not an actual response')
            item.setdefault('requested', {'model': None, 'reasoning_effort': None})
        if kind == 'execution':
            _require(item.get('request_id') and index[item['request_id']]['kind'] == 'request',
                     'execution requires request')
            item.setdefault('observed', {})
            for key in ('model', 'reasoning_effort'):
                item['observed'].setdefault(key, None)
            if any(v is not None for v in item['observed'].values()):
                _require(item.get('host_receipt') and item['evidence'],
                         'actual model metadata requires host receipt evidence')
            if item.get('status') in ('completed', 'responded'):
                _require(item.get('host_task_handle') and item['evidence'],
                         'completed execution requires actual invocation and response evidence')
        if kind == 'review':
            execution = index.get(item.get('execution_id'), {})
            _require(execution.get('kind') == 'execution' and execution.get('status') in ('completed', 'responded'),
                     'review requires completed actual execution')
            _require(item.get('reviewer') and item.get('author')
                     and item['reviewer'] != item['author'], 'independent review cannot be self-review')
            _require(item.get('refs') and item.get('independent') is True,
                     'review requires pinned target and explicit independence')
            authorship = [index[r['id']] for r in item.get('refs', [])]
            _require(all(r.get('owner') != item['reviewer'] and r.get('author') != item['reviewer']
                         for r in authorship), 'reviewer authored target')
            _require(all(r.get('execution_id') != item['execution_id'] for r in authorship),
                     'review requires a separate invocation')
        if kind in ('task', 'stage') and item.get('status') == 'completed':
            _require(item['evidence'] and item.get('completion_conditions'),
                     'completion requires conditions and evidence')
        if kind == 'stage':
            _require(item.get('stage_id') and item.get('owner') and item.get('reason'),
                     'stage requires stage_id, owner, reason')
            if item.get('status') == 'completed':
                _require(item.get('refs') or item.get('depends_on'),
                         'completed stage requires pinned task/decision dependencies')
            if item.get('supersedes'):
                previous = index[item['supersedes']]
                _require(previous.get('stage_id') == item['stage_id'], 'stage supersedes wrong stage')
                if item.get('status') == 'completed':
                    _require(item.get('change_id') in index and index[item['change_id']]['kind'] == 'change',
                             'stage reassessment requires change reference')
                    _require({e['sha256'] for e in item['evidence']} -
                             {e['sha256'] for e in previous.get('evidence', [])},
                             'reassessment needs new evidence, not a version relabel')
        if kind == 'change':
            _require(item.get('targets') and all(x in index for x in item['targets']),
                     'change requires explicit known targets')
            _require(item.get('reason') and item.get('old_version') and item.get('new_version')
                     and item['old_version'] != item['new_version'] and item['evidence'],
                     'change requires distinct old/new versions, reason and evidence')
        if kind in ('time', 'session'):
            for interval in item.get('intervals', []):
                _require(_stamp(interval['end']) >= _stamp(interval['start']), 'reversed interval')
            if item.get('intervals'):
                _require(item.get('source'), 'observed intervals require source')
                _require(item.get('actor') if item.get('unit') == 'person_seconds' else item.get('resource'),
                         'observed intervals require actor/resource')
        if kind == 'reservation':
            _require(item.get('task_id') and item.get('unit'), 'reservation requires task and unit')
            for key in ('planned', 'limit'):
                if item.get(key) is not None:
                    _number(item[key], key)
        if kind == 'receipt':
            _require(item.get('state') in ('local_verified', 'upload_requested', 'remote_verified',
                                          'download_verified', 'restore_verified'), 'unknown receipt state')
            if item['state'] != 'upload_requested':
                _require(item['evidence'] and item.get('verified_at') and item.get('method'),
                         'verified receipt requires file evidence, timestamp and method')
                _stamp(item['verified_at'])
            if item['state'] in ('remote_verified', 'download_verified', 'restore_verified'):
                _require(item.get('destination') and item.get('verification_evidence'),
                         'remote verification requires destination and observed verification evidence')
                item['verification_evidence'] = _evidence(item['verification_evidence'], record_path.parent)
            if item['state'] == 'restore_verified':
                _require(item.get('restore_result') == 'passed' and item.get('new_session_id'),
                         'restore requires passed execution in a new session')
        item['sha256'] = hash_json(item)
        ledger['records'].append(item)
        _atomic(ledger_path, ledger)
        return item


def _union(intervals):
    total = 0
    end = None
    for left, right in sorted(intervals):
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def status(ledger_path):
    """Summarize historical execution separately from current evidence validity."""
    records = _ledger(ledger_path)['records']
    index = {r['id']: r for r in records}
    affected = {}
    latest_stages = {}
    intervals = defaultdict(list)
    task_intervals = defaultdict(list)
    latest_reservations = {}
    for item in records:
        rid = item['id']
        if item['kind'] == 'change':
            impacted = set(item['targets'])
            while True:
                expanded = impacted | {r['id'] for r in records if any(
                    ref['id'] in impacted for ref in r.get('refs', []) + r.get('depends_on', []))}
                if expanded == impacted:
                    break
                impacted = expanded
            for target in impacted:
                affected[target] = rid
        if item['kind'] == 'stage':
            latest_stages[item['stage_id']] = item
        if item['kind'] in ('time', 'session'):
            unit = item.get('unit')
            if unit not in ('person_seconds', 'gpu_seconds', 'execution_seconds'):
                continue
            identity = item.get('actor') if unit == 'person_seconds' else item.get('resource')
            for interval in item.get('intervals', []):
                if interval.get('observed', True) is False:
                    continue
                pair = (_stamp(interval['start']), _stamp(interval['end']))
                intervals[(unit, identity)].append(pair)
                if item.get('task_id'):
                    task_intervals[(item['task_id'], unit, identity)].append(pair)
        if item['kind'] == 'reservation':
            latest_reservations[(item['task_id'], item['unit'])] = item
    stages = {}
    for key, item in latest_stages.items():
        change = affected.get(item['id'])
        stages[key] = {'record_id': item['id'], 'historical_status': item.get('status'),
                       'validity': 'needs_reassessment' if change else item.get('validity', 'current'),
                       'change_id': change, 'next_action': item.get('next_action')}
    evidence_status = {}
    for item in records:
        for entry in item.get('evidence', []) + item.get('verification_evidence', []):
            path = Path(entry['path'])
            if not path.is_file() or hash_file(path) != entry['sha256']:
                evidence_status.setdefault(item['id'], []).append(entry['path'])
    for item in latest_stages.values():
        pending = [item['id']]
        visited = set()
        stale = []
        while pending:
            rid = pending.pop()
            if rid in visited:
                continue
            visited.add(rid)
            stale.extend(evidence_status.get(rid, []))
            pending.extend(r['id'] for r in index[rid].get('refs', []) + index[rid].get('depends_on', []))
        if stale:
            stages[item['stage_id']]['validity'] = 'needs_reassessment'
            stages[item['stage_id']]['stale_evidence'] = sorted(set(stale))
    actual = defaultdict(float)
    for (unit, _), spans in intervals.items():
        actual[unit] += _union(spans)
    budgets = {}
    for (task, unit), reservation in latest_reservations.items():
        used = sum(_union(spans) for (tid, u, _), spans in task_intervals.items() if tid == task and u == unit)
        planned = reservation.get('planned')
        current_task = index[task]
        for candidate in records:
            if candidate.get('supersedes') == current_task['id'] and candidate['kind'] == 'task':
                current_task = candidate
        finished = current_task.get('status') in ('completed', 'cancelled', 'failed')
        remaining = 0 if finished or reservation.get('status') in ('released', 'cancelled') else (
            None if planned is None else max(0, planned - used))
        budget = budgets.setdefault(unit, {'actual': actual.get(unit, 0), 'reserved_remaining': 0,
                                          'limit': None, 'reservations': []})
        if remaining is None or budget['reserved_remaining'] is None:
            budget['reserved_remaining'] = None
        else:
            budget['reserved_remaining'] += remaining
        if reservation.get('limit') is not None:
            budget['limit'] = reservation['limit']
        budget['reservations'].append({'id': reservation['id'], 'task_id': task, 'actual': used,
                                       'reserved_remaining': remaining})
    for budget in budgets.values():
        budget['within_limit'] = (None if budget['limit'] is None or budget['reserved_remaining'] is None
                                  else budget['actual'] + budget['reserved_remaining'] <= budget['limit'])
    return {'records': len(records), 'kinds': dict(Counter(r['kind'] for r in records)),
            'stages': stages, 'needs_reassessment': affected, 'actual': dict(actual), 'budgets': budgets,
            'evidence_freshness': {'fresh': not evidence_status, 'stale_records': evidence_status},
            'colab_quota': 'unknown', 'sessions': [dict(r, unobserved_tail=r.get('status') != 'observed_closed')
                for r in records if r['kind'] == 'session'],
            'limitations': ['Only explicit dependencies are propagated; unrecorded relationships remain unknown.',
                            'Unknown session tails are excluded from observed resource time.']}


def results(run_paths, output_dir):
    """Export recorded run outcomes; never manufacture metrics for absent runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for entry in run_paths:
        path = Path(entry)
        if path.is_dir():
            candidates = [path / name for name in ('result.json', 'run.json', 'summary.json') if (path / name).is_file()]
            _require(candidates, f'no run result in {path}')
            path = candidates[0]
        value = load_json(path)
        outcome = value.get('status')
        _require(outcome in ('completed', 'failed', 'interrupted'), 'run result must have an actual outcome')
        metrics = value.get('metrics', {})
        _require(isinstance(metrics, dict), 'metrics must be a mapping')
        for metric, number in metrics.items():
            _require(not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number),
                     f'non-finite metric: {metric}')
        runs.append({'run_id': value.get('run_id', value.get('id', path.parent.name)), 'status': outcome,
                     'metrics': metrics, 'source': str(path.resolve()), 'sha256': hash_file(path),
                     'protocol': value.get('protocol_hash', value.get('protocol')), 'split_hash': value.get('split_hash'),
                     'failure': value.get('failure', value.get('error'))})
    _require(len({r['run_id'] for r in runs}) == len(runs), 'duplicate run ID')
    keys = sorted({key for run in runs for key in run['metrics']})
    csv_path = output_dir / 'results.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run_id', 'status', 'protocol', 'split_hash', 'failure'] + keys)
        writer.writeheader()
        for run in runs:
            writer.writerow({**{k: run[k] for k in ('run_id', 'status', 'protocol', 'split_hash', 'failure')}, **run['metrics']})
    summary = {'runs': runs, 'counts': dict(Counter(r['status'] for r in runs)),
               'csv_path': str(csv_path.resolve()), 'wandb_uploaded': False,
               'limitations': ['Recorded outcomes only; different split/protocol results are not pooled.']}
    _atomic(output_dir / 'report-data.json', summary)
    return summary


def _claim_dependencies(claim, brief, index, stage_status):
    """Follow declared evidence/review links, without inventing section dependencies."""
    pending = [ref['id'] for field in ('refs', 'depends_on') for ref in claim.get(field, [])]
    if claim.get('review_id'):
        pending.append(claim['review_id'])
    for stage_id in brief.get('stage_ids', []) + claim.get('stage_ids', []):
        _require(stage_id in stage_status, f'unknown stage dependency: {stage_id}')
        pending.append(stage_status[stage_id]['record_id'])
    visited = set()
    while pending:
        rid = pending.pop()
        if rid in visited:
            continue
        visited.add(rid)
        item = index.get(rid)
        if item is None:
            # Missing review IDs are rejected by final-review validation, not
            # treated as proof of a current independent review here.
            continue
        pending.extend(ref['id'] for field in ('refs', 'depends_on') for ref in item.get(field, []))
        pending.extend(item[field] for field in ('task_id', 'request_id', 'execution_id') if item.get(field))
    return visited


def _affected_claim_records(claim, brief, index, assessment):
    affected = set(assessment['needs_reassessment'])
    affected.update(r['record_id'] for r in assessment['stages'].values()
                    if r['validity'] == 'needs_reassessment')
    affected.update(assessment['evidence_freshness']['stale_records'])
    return sorted(_claim_dependencies(claim, brief, index, assessment['stages']) & affected)


def context(ledger_path, brief_path, output_path):
    """Snapshot a section brief and its explicit claims, sources and limitations.

    Brief: section, purpose, claims [{claim_id,text,kind,evidence,refs,status}],
    evidence [{path,sha256}], limitations (nonempty). No prose is generated.
    """
    ledger = _ledger(ledger_path)
    brief = load_json(brief_path)
    assessment = status(ledger_path)
    _require(brief.get('section') and brief.get('purpose') and brief.get('limitations'),
             'brief requires section, purpose, and explicit limitations')
    index = {r['id']: r for r in ledger['records']}
    sources = _evidence(brief.get('evidence', []), Path(brief_path).parent)
    claims = []
    for original in brief.get('claims', []):
        claim = dict(original)
        _require(claim.get('claim_id') and claim.get('text') and claim.get('kind') in
                 ('external', 'observation', 'interpretation', 'hypothesis', 'plan'), 'invalid claim')
        _references(claim, index)
        claim['evidence'] = _evidence(claim.get('evidence', []), Path(brief_path).parent)
        pinned = [index[r['id']] for r in claim.get('refs', [])]
        pending = list(pinned)
        visited = set()
        while pending:
            item = pending.pop()
            if item['id'] in visited:
                continue
            visited.add(item['id'])
            sources.extend(_evidence(item.get('evidence', []), Path(brief_path).parent))
            pending.extend(index[r['id']] for r in item.get('refs', []) + item.get('depends_on', []))
        if not claim['evidence'] and not pinned:
            claim['status'] = 'needs_evidence'
        else:
            claim.setdefault('status', 'evidence_linked')
        affected = _affected_claim_records(claim, brief, index, assessment)
        if affected:
            claim['status'] = 'needs_update'
            claim['affected_records'] = affected
        claims.append(claim)
        sources.extend(claim['evidence'])
    _require(len({c['claim_id'] for c in claims}) == len(claims), 'duplicate claim ID')
    sources.extend([{'path': str(Path(ledger_path).resolve()), 'sha256': hash_file(ledger_path)},
                    {'path': str(Path(brief_path).resolve()), 'sha256': hash_file(brief_path)}])
    package = {'schema_version': 1, 'section': brief['section'], 'purpose': brief['purpose'],
               'claims': claims, 'brief': brief, 'sources': list({(s['path'], s['sha256']): s for s in sources}.values()),
               'ledger_records': ledger['records'], 'ledger_path': str(Path(ledger_path).resolve()),
               'stage_status': assessment['stages'], 'needs_reassessment': assessment['needs_reassessment'],
               'limitations': brief['limitations'], 'review_executed': False, 'frozen': False}
    _atomic(output_path, package)
    return package


def freshness(context_path):
    package = load_json(context_path)
    changed, missing = [], []
    for source in package.get('sources', []):
        path = Path(source['path'])
        if not path.is_file():
            missing.append(str(path))
        elif hash_file(path) != source['sha256']:
            changed.append(str(path))
    return {'fresh': not changed and not missing, 'changed': changed, 'missing': missing}


def freeze(context_path, output_path):
    package = load_json(context_path)
    _require(freshness(context_path)['fresh'], 'cannot freeze changed or missing evidence')
    _require(all(c.get('status') not in ('needs_update', 'needs_reassessment')
                 for c in package.get('claims', [])), 'cannot freeze claims requiring reassessment or update')
    _require(package.get('ledger_path'), 'context must identify its evidence ledger')
    assessment = status(package['ledger_path'])
    index = {r['id']: r for r in package['ledger_records']}
    _require(all(not _affected_claim_records(c, package.get('brief', {}), index, assessment)
                 for c in package.get('claims', [])),
             'cannot freeze claims referencing evidence requiring reassessment')
    _require(all(c.get('status') != 'needs_evidence' for c in package.get('claims', [])),
             'cannot freeze claims missing evidence')
    if package.get('brief', {}).get('final'):
        _require(all(c.get('status') == 'reviewed' and c.get('review_id') for c in package.get('claims', [])),
                 'final claims require independent review records')
        records = {r['id']: r for r in package['ledger_records']}
        for claim in package['claims']:
            review = records.get(claim['review_id'], {})
            _require(review.get('kind') == 'review' and review.get('independent') is True,
                     'final claim review not found')
            covered_evidence = {e['sha256'] for ref in review.get('refs', [])
                                for e in records[ref['id']].get('evidence', [])}
            _require({e['sha256'] for e in claim.get('evidence', [])} <= covered_evidence,
                     'review does not cover direct claim evidence')
            _require({r['id'] for r in claim.get('refs', [])} <= {r['id'] for r in review.get('refs', [])}
                     and claim.get('refs'), 'review does not cover claim evidence versions')
    package['frozen'] = True
    package['context_sha256'] = hash_file(context_path)
    _atomic(output_path, package)
    return package
