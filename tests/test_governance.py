"""Evidence integrity and current-vs-historical governance behavior."""
import json
import tempfile
import unittest
from pathlib import Path

from defectlab.common import hash_file
from defectlab import governance as gov


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / 'ledger.json'
        self.counter = 0

    def file(self, name, text):
        path = self.root / name
        path.write_text(text, encoding='utf-8')
        return {'path': str(path), 'sha256': hash_file(path)}

    def add(self, **item):
        self.counter += 1
        path = self.root / f'record-{self.counter}.json'
        path.write_text(json.dumps(item), encoding='utf-8')
        return gov.record(self.ledger, path)

    def ref(self, record):
        return {'id': record['id'], 'sha256': record['sha256']}

    def task(self):
        return self.add(id='task', kind='task', owner='writer', status='in_progress')

    def test_atomic_validation_and_pinned_references(self):
        task = self.task()
        previous = self.ledger.read_bytes()
        with self.assertRaises(ValueError):
            self.add(id='task', kind='task')
        with self.assertRaises(ValueError):
            self.add(id='bad', kind='decision', refs=[{'id': 'task', 'sha256': 'wrong'}])
        self.assertEqual(previous, self.ledger.read_bytes())
        decision = self.add(id='decision', kind='decision', refs=[self.ref(task)])
        self.assertEqual(len(decision['sha256']), 64)
        payload = json.loads(self.ledger.read_text())
        payload['records'][0]['owner'] = 'tampered'
        self.ledger.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            gov.status(self.ledger)

    def test_requests_are_not_responses_and_review_is_independent(self):
        task = self.task()
        self.add(id='request', kind='request', task_id='task', requested={'model': 'desired'})
        execution = self.add(id='execution', kind='execution', request_id='request',
                             status='completed', host_task_handle='actual-call-1',
                             evidence=[self.file('response.txt', 'Response from separate call')])
        self.assertEqual(execution['observed'], {'model': None, 'reasoning_effort': None})
        with self.assertRaisesRegex(ValueError, 'self-review'):
            self.add(id='self', kind='review', execution_id='execution', independent=True,
                     author='writer', reviewer='writer', refs=[self.ref(task)])
        with self.assertRaisesRegex(ValueError, 'authored'):
            self.add(id='spoof', kind='review', execution_id='execution', independent=True,
                     author='someone', reviewer='writer', refs=[self.ref(task)])
        review = self.add(id='review', kind='review', execution_id='execution', independent=True,
                         author='writer', reviewer='peer', refs=[self.ref(task)])
        self.assertEqual(review['kind'], 'review')
        with self.assertRaisesRegex(ValueError, 'actual response'):
            self.add(id='prepared', kind='request', status='reviewed')
        with self.assertRaisesRegex(ValueError, 'host receipt'):
            self.add(id='invented', kind='execution', request_id='request', observed={'model': 'guessed'})

    def test_fh01_changes_invalidate_transitive_stages_preserve_completed_history(self):
        old = self.file('split-v1.json', 'split v1')
        task = self.add(id='split', kind='task', status='completed', completion_conditions=['fixed split'],
                        evidence=[old])
        stage = self.add(id='A2-v1', kind='stage', stage_id='A2', status='completed', owner='conductor',
                         reason='conditions verified', completion_conditions=['split fixed'],
                         refs=[self.ref(task)], evidence=[old])
        self.add(id='B-v1', kind='stage', stage_id='B', status='completed', owner='conductor',
                 reason='baseline observed', completion_conditions=['baseline'],
                 depends_on=[self.ref(stage)], evidence=[self.file('run.json', 'completed run')])
        new = self.file('split-v2.json', 'split v2')
        self.add(id='change', kind='change', targets=['split'], old_version=old['sha256'],
                 new_version=new['sha256'], evidence=[new], reason='group correction')
        state = gov.status(self.ledger)
        self.assertEqual(state['stages']['A2']['validity'], 'needs_reassessment')
        self.assertEqual(state['stages']['B']['validity'], 'needs_reassessment')
        self.assertEqual(state['stages']['B']['historical_status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'new evidence'):
            self.add(id='A2-fake', kind='stage', stage_id='A2', status='completed', owner='conductor',
                     reason='relabel', completion_conditions=['split fixed'], supersedes='A2-v1',
                     change_id='change', evidence=[old], refs=[self.ref(task)])
        newtask = self.add(id='split-v2', kind='task', status='completed', completion_conditions=['fixed split'],
                           evidence=[new], supersedes='split')
        self.add(id='A2-v2', kind='stage', stage_id='A2', status='completed', owner='conductor',
                 reason='new split verified', completion_conditions=['split fixed'], supersedes='A2-v1',
                 change_id='change', evidence=[new], refs=[self.ref(newtask)])
        state = gov.status(self.ledger)
        self.assertEqual(state['stages']['A2']['validity'], 'current')
        self.assertEqual(state['stages']['B']['validity'], 'needs_reassessment')

    def test_time_unions_resource_separation_reservation_consumption_unknown_tail(self):
        self.task()
        self.add(id='reserve', kind='reservation', task_id='task', unit='gpu_seconds', planned=200, limit=250)
        for rid, start, end in [('gpu1', 0, 60), ('gpu2', 30, 90)]:
            self.add(id=rid, kind='session', task_id='task', unit='gpu_seconds', resource='T4-1',
                     status='end_unknown', source='manual telemetry', intervals=[{
                         'start': f'2026-11-17T00:0{start // 60}:{start % 60:02d}+00:00',
                         'end': f'2026-11-17T00:0{end // 60}:{end % 60:02d}+00:00'}])
        self.add(id='unknown', kind='session', task_id='task', unit='gpu_seconds', resource='T4-1',
                 status='end_unknown', source='manual', intervals=[{
                     'start': '2026-11-17T00:01:30+00:00', 'end': '2026-11-17T01:00:00+00:00',
                     'observed': False}])
        self.add(id='person', kind='time', actor='researcher', unit='person_seconds', source='manual',
                 intervals=[{'start': '2026-11-17T00:00:00+00:00', 'end': '2026-11-17T00:01:00+00:00'}])
        state = gov.status(self.ledger)
        self.assertEqual(state['actual'], {'gpu_seconds': 90, 'person_seconds': 60})
        self.assertEqual(state['budgets']['gpu_seconds']['reserved_remaining'], 110)
        self.assertTrue(state['budgets']['gpu_seconds']['within_limit'])
        self.assertEqual(state['colab_quota'], 'unknown')
        self.assertTrue(state['sessions'][0]['unobserved_tail'])
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.add(id='negative', kind='reservation', task_id='task', unit='gpu_seconds', planned=-1)

    def test_receipts_require_actual_local_hash_and_restore_evidence(self):
        artifact = self.file('weights.bin', 'weight bytes')
        self.add(id='upload', kind='receipt', state='upload_requested')
        with self.assertRaisesRegex(ValueError, 'destination'):
            self.add(id='durable', kind='receipt', state='remote_verified', evidence=[artifact],
                     verified_at='2026-11-17T00:00:00Z', method='upload requested')
        proof = self.file('download-receipt.json', '{"download_hash_matches":true}')
        self.add(id='restore', kind='receipt', state='restore_verified', evidence=[artifact],
                 verified_at='2026-11-17T00:00:00Z', method='download and resume', destination='drive://actual-file',
                 verification_evidence=[proof], restore_result='passed', new_session_id='colab-2')
        Path(artifact['path']).write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.add(id='corrupt', kind='receipt', state='local_verified', evidence=[artifact],
                     verified_at='2026-11-17T00:00:00Z', method='hash')

    def test_real_result_export_context_freshness_and_freeze(self):
        task = self.task()
        for name, status in [('run-good', 'completed'), ('run-bad', 'failed')]:
            self.file(name + '.json', json.dumps({'run_id': name, 'status': status,
                      'metrics': {'auroc': .75} if status == 'completed' else {},
                      'protocol': 'holdout-v1', 'split_hash': 'fixed'}))
        exported = gov.results([self.root / 'run-good.json', self.root / 'run-bad.json'], self.root / 'tables')
        self.assertEqual(exported['counts'], {'completed': 1, 'failed': 1})
        self.assertFalse(exported['wandb_uploaded'])
        source = self.file('source.txt', 'Observed baseline result')
        brief = self.root / 'brief.json'
        brief.write_text(json.dumps({'section': 'Results', 'purpose': 'Describe observed run',
            'limitations': ['One holdout; no private labels'], 'claims': [
                {'claim_id': 'C1', 'text': 'Baseline observed', 'kind': 'observation',
                 'evidence': [source], 'refs': [self.ref(task)]}]}))
        context_path = self.root / 'context.json'
        package = gov.context(self.ledger, brief, context_path)
        self.assertEqual(package['claims'][0]['status'], 'evidence_linked')
        self.assertTrue(gov.freshness(context_path)['fresh'])
        frozen = gov.freeze(context_path, self.root / 'frozen.json')
        self.assertTrue(frozen['frozen'])
        Path(source['path']).write_text('Changed')
        self.assertFalse(gov.freshness(context_path)['fresh'])
        with self.assertRaisesRegex(ValueError, 'changed'):
            gov.freeze(context_path, self.root / 'frozen-2.json')

    def test_raw_evidence_change_invalidates_stage_and_blocks_new_context(self):
        source = self.file('evidence.txt', 'original')
        task = self.add(id='task', kind='task', status='completed',
                        completion_conditions=['observed'], evidence=[source])
        self.add(id='stage', kind='stage', stage_id='A', owner='conductor', reason='observed',
                 status='completed', completion_conditions=['done'], evidence=[source], refs=[self.ref(task)])
        Path(source['path']).write_text('new bytes')
        state = gov.status(self.ledger)
        self.assertEqual(state['stages']['A']['validity'], 'needs_reassessment')
        self.assertFalse(state['evidence_freshness']['fresh'])
        brief = self.root / 'brief.json'
        brief.write_text(json.dumps({'section': 'Methods', 'purpose': 'Snapshot actual evidence',
            'limitations': ['Only recorded facts'], 'claims': [{'claim_id': 'C', 'text': 'Observed',
            'kind': 'observation', 'refs': [self.ref(task)]}]}))
        with self.assertRaisesRegex(ValueError, 'changed'):
            gov.context(self.ledger, brief, self.root / 'context.json')

    def test_reviewed_final_freeze_binds_claim_to_exact_reviewed_evidence(self):
        source = self.file('evidence.txt', 'real completed result')
        task = self.add(id='task', kind='task', owner='writer', status='completed',
                        completion_conditions=['observed'], evidence=[source])
        self.add(id='request', kind='request', task_id='task')
        self.add(id='execution', kind='execution', request_id='request', status='completed',
                 host_task_handle='peer-call', evidence=[self.file('response.txt', 'checked evidence')])
        self.add(id='review', kind='review', execution_id='execution', author='writer', reviewer='peer',
                 independent=True, refs=[self.ref(task)])
        brief = self.root / 'brief.json'
        definition = {'section': 'Conclusion', 'purpose': 'Final observed claim', 'final': True,
            'limitations': ['One holdout'], 'claims': [{'claim_id': 'C', 'text': 'Observed result',
            'kind': 'observation', 'status': 'reviewed', 'review_id': 'review',
            'refs': [self.ref(task)], 'evidence': [source]}]}
        brief.write_text(json.dumps(definition))
        path = self.root / 'context.json'
        gov.context(self.ledger, brief, path)
        self.assertTrue(gov.freeze(path, self.root / 'frozen.json')['frozen'])
        definition['claims'][0]['evidence'] = [self.file('unreviewed.txt', 'unreviewed result')]
        brief.write_text(json.dumps(definition))
        gov.context(self.ledger, brief, path)
        with self.assertRaisesRegex(ValueError, 'direct claim evidence'):
            gov.freeze(path, self.root / 'fake-final.json')

    def test_explicit_change_blocks_old_reviewed_final_even_when_old_bytes_are_preserved(self):
        old = self.file('split-v1.json', 'unchanged historical split v1')
        task = self.add(id='task', kind='task', owner='writer', status='completed',
                        completion_conditions=['completed under v1'], evidence=[old])
        stage = self.add(id='stage-C', kind='stage', stage_id='C', owner='conductor',
                         reason='observed baseline', status='completed', completion_conditions=['baseline'],
                         evidence=[old], refs=[self.ref(task)])
        self.add(id='request', kind='request', task_id='task')
        self.add(id='execution', kind='execution', request_id='request', status='completed',
                 host_task_handle='peer-call', evidence=[self.file('response.txt', 'reviewed v1 baseline')])
        self.add(id='review', kind='review', execution_id='execution', author='writer', reviewer='peer',
                 independent=True, refs=[self.ref(task)])
        new = self.file('split-v2.json', 'replacement split v2')
        self.add(id='change', kind='change', targets=['task'], old_version=old['sha256'],
                 new_version=new['sha256'], evidence=[new], reason='group membership corrected')
        self.assertEqual(hash_file(old['path']), old['sha256'])
        state = gov.status(self.ledger)
        self.assertEqual(state['stages']['C']['validity'], 'needs_reassessment')
        self.assertEqual(state['stages']['C']['historical_status'], 'completed')
        definition = {'section': 'Conclusion', 'purpose': 'Final observed claim', 'final': True,
            'limitations': ['One holdout'], 'claims': [{'claim_id': 'C1', 'text': 'Old baseline result',
            'kind': 'observation', 'status': 'reviewed', 'review_id': 'review',
            'refs': [self.ref(task)], 'evidence': [old]}]}
        brief = self.root / 'brief.json'
        brief.write_text(json.dumps(definition))
        path = self.root / 'context.json'
        package = gov.context(self.ledger, brief, path)
        self.assertTrue(gov.freshness(path)['fresh'])
        self.assertEqual(package['claims'][0]['status'], 'needs_update')
        self.assertIn('task', package['claims'][0]['affected_records'])
        self.assertIn('review', package['claims'][0]['affected_records'])
        with self.assertRaisesRegex(ValueError, 'reassessment'):
            gov.freeze(path, self.root / 'invalid-final.json')
        # Final freeze independently checks graph validity, even if someone
        # attempts to restore the historical claim's old reviewed label.
        package['claims'][0]['status'] = 'reviewed'
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, 'reassessment'):
            gov.freeze(path, self.root / 'invalid-final-2.json')
        # A claim naming only the downstream stage is also affected transitively.
        definition['claims'][0]['refs'] = [self.ref(stage)]
        brief.write_text(json.dumps(definition))
        package = gov.context(self.ledger, brief, path)
        self.assertEqual(package['claims'][0]['status'], 'needs_update')
        # A declared section-stage dependency applies even to a different claim.
        unrelated = self.add(id='unrelated', kind='task', evidence=[self.file('other.txt', 'independent')])
        definition['stage_ids'] = ['C']
        definition['claims'][0]['refs'] = [self.ref(unrelated)]
        definition['claims'][0].pop('review_id')
        brief.write_text(json.dumps(definition))
        package = gov.context(self.ledger, brief, path)
        self.assertIn('stage-C', package['claims'][0]['affected_records'])
        # Unrelated writing work remains possible when it has no declared stale dependency.
        definition.pop('stage_ids')
        definition['claims'][0]['status'] = 'evidence_linked'
        brief.write_text(json.dumps(definition))
        package = gov.context(self.ledger, brief, path)
        self.assertEqual(package['claims'][0]['status'], 'evidence_linked')

    def test_final_context_cannot_inherit_unverified_review_status(self):
        self.task()
        source = self.file('source.txt', 'Actual observation')
        brief = self.root / 'brief.json'
        brief.write_text(json.dumps({'section': 'Conclusion', 'purpose': 'Final claims', 'final': True,
            'limitations': ['Limited test'], 'claims': [{'claim_id': 'C1', 'text': 'Result',
            'kind': 'observation', 'status': 'reviewed', 'review_id': 'fake', 'evidence': [source]}]}))
        path = self.root / 'context.json'
        gov.context(self.ledger, brief, path)
        with self.assertRaisesRegex(ValueError, 'review not found'):
            gov.freeze(path, self.root / 'final.json')


if __name__ == '__main__':
    unittest.main()
