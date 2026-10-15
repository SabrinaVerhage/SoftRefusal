"""CPU-only reference regressions. Run with unittest; only NumPy is required.

Export tests stub UMAP deliberately: they exercise frozen-donut persistence,
not UMAP quality or a live model.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from avoidance.mapping import export_map, fit_semantic_reference, semantic_layout


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.x = np.random.default_rng(13).normal(size=(8, 6))
        self.labels = ['answered', 'refused', 'answered', 'refused', 'redirected', 'unclear', 'unlabeled', 'answered']

    def tearDown(self):
        self.temp.cleanup()

    def session(self, size=6, labels=None, dtype='bfloat16', template='template-a'):
        directory = self.root / 'runs' / 'session-a'
        checkpoint = directory / 'checkpoints' / str(size)
        checkpoint.mkdir(parents=True, exist_ok=True)
        rows = [dict(point_id=f'p{i}', conversation_id='a', turn_id=i, activation_row=i,
                     prompt=f'Neutral fixture {i}', response='Fixture response') for i in range(size)]
        (checkpoint / 'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        np.savez(checkpoint / 'activations.npz', layer_0=self.x[:size], layer_1=self.x[:size] * 2)
        manifest = dict(status='complete', synthetic=False, site='resid_post', module_path='fixture.layers',
                        chat_template_sha256=template, environment={'packages': {'transformers': 'fixture'}},
                        config=dict(model_id='fixture-only', revision='fixture', layers=[0, 1],
                                    adapter='remote', dtype=dtype, token_selection='last_prompt'))
        manifest['hashes'] = {n: hashlib.sha256((checkpoint / n).read_bytes()).hexdigest()
                              for n in ('steps.jsonl', 'activations.npz')}
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
        (directory / 'session.json').write_text(json.dumps(dict(created_utc='2026-01-01', latest=f'checkpoints/{size}', title='Fixture')))
        (directory / 'labels.jsonl').write_text('\n'.join(json.dumps(dict(point_id=f'p{i}', outcome=label))
                                                        for i, label in enumerate(labels or self.labels[:size])))

    def export(self, **kwargs):
        with patch('avoidance.mapping.fit_layer', side_effect=lambda x, *a: (x[:, :3], {'fixture': True})):
            export_map(self.root / 'runs', self.root / 'map.js', **kwargs)
        return json.loads((self.root / 'map.json').read_text())

    @staticmethod
    def positions(payload):
        return {p['point_id']: [p[k] for k in ('core_score', 'angle', 'lift')] for p in payload['points']}

    def test_append_and_relabel_leave_previous_positions_unchanged(self):
        self.session()
        original = self.export()
        ref = (self.root / 'map.reference.json').read_bytes()
        self.session(8, ['refused', 'answered'] + ['unlabeled'] * 6)
        updated = self.export()
        for key, position in self.positions(original).items():
            self.assertEqual(position, self.positions(updated)[key])
        self.assertEqual(ref, (self.root / 'map.reference.json').read_bytes())
        self.assertEqual(updated['points'][0]['outcome'], 'refused')
        self.assertEqual(original['semantic']['reference_id'], updated['semantic']['reference_id'])

    def test_automatic_labels_define_reference_and_human_corrections_win(self):
        self.session()
        directory = self.root / 'runs' / 'session-a'
        (directory / 'labels.jsonl').write_text('')
        (directory / 'auto_labels.jsonl').write_text('\n'.join(json.dumps(dict(point_id=f'p{i}', outcome=v,
             status='complete', judge_version='fixture', model_id='fixture', revision='fixture', reason='Fixture judgment'))
             for i, v in enumerate(self.labels[:6])))
        original = self.export()
        self.assertEqual(original['points'][0]['label_source'], 'model_judge')
        self.assertEqual(original['semantic']['reference_label_sources']['model_judge'], 4)
        (directory / 'labels.jsonl').write_text(json.dumps(dict(point_id='p0',outcome='unclear',label_source='human')))
        corrected = self.export()
        self.assertEqual(corrected['points'][0]['outcome'], 'unclear')
        self.assertEqual(corrected['points'][0]['label_source'], 'human')
        self.assertEqual(self.positions(original), self.positions(corrected))

    def test_explicit_rebuild_archives_reference(self):
        self.session()
        old = self.export()
        old_bytes = (self.root / 'map.reference.json').read_bytes()
        self.session(8)
        new = self.export(rebuild_reference=True)
        self.assertNotEqual(old['semantic']['reference_id'], new['semantic']['reference_id'])
        archive = self.root / f"map.reference-{old['semantic']['reference_id']}.json"
        self.assertEqual(old_bytes, archive.read_bytes())

    def test_failed_rebuild_preserves_all_outputs(self):
        self.session()
        self.export()
        names = ['map.reference.json', 'map.js', 'map.json']
        before = {n: (self.root / n).read_bytes() for n in names}
        self.session(labels=['answered'] * 6)
        with self.assertRaisesRegex(ValueError, 'Cannot rebuild'):
            self.export(rebuild_reference=True)
        self.assertEqual(before, {n: (self.root / n).read_bytes() for n in names})

    def test_measurement_changes_rejected(self):
        self.session()
        self.export()
        for kwargs in [dict(dtype='float32'), dict(template='different-template')]:
            self.session(**kwargs)
            with self.assertRaisesRegex(ValueError, 'different measurement'):
                self.export()

    def test_layer_change_requires_explicit_rebuild(self):
        self.session()
        self.export()
        with self.assertRaisesRegex(ValueError, 'different measurement'):
            self.export(layer=1)

    def test_redirection_cannot_substitute_for_refusal(self):
        self.session(labels=['answered', 'redirected'] * 3)
        result = self.export()
        self.assertIn('unavailable', result['semantic'])
        self.assertFalse((self.root / 'map.reference.json').exists())

    def test_identical_group_means_are_handled(self):
        x = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
        result, reason = semantic_layout(x, ['answered'] * 2 + ['refused'] * 2)
        self.assertIsNone(result)
        self.assertIn('no distinct direction', reason)

    def test_zero_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, 'zero vectors'):
            semantic_layout(np.zeros((2, 3)), ['answered', 'refused'])

    def test_reference_group_means_and_independent_transform(self):
        ref, _ = fit_semantic_reference(self.x, self.labels)
        result, _ = semantic_layout(self.x, self.labels, ref)
        for outcome, expected in [('answered', 0), ('refused', 1)]:
            self.assertAlmostEqual(result['score'][np.array(self.labels) == outcome].mean(), expected)
        # Scoring one point must not recenter/reorient/rescale against its batch.
        for i in range(len(self.x)):
            single, _ = semantic_layout(self.x[i:i+1], [self.labels[i]], ref)
            for key in ('score', 'angle', 'lift'):
                self.assertAlmostEqual(single[key][0], result[key][i])


if __name__ == '__main__':
    unittest.main()
