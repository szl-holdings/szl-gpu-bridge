"""Offline, synthetic Windows memory observation regression tests."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools' / 'evidence' / 'windows_memory_envelope.py'
spec = importlib.util.spec_from_file_location('szl_windows_memory_envelope', SOURCE)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(n=5, free_gib=8, commit_ppm=700000, pagefile_gib=12):
    total = 32 * m.GIB
    limit = 64 * m.GIB
    base = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            'observed_utc': (base + timedelta(seconds=2*i)).isoformat(),
            'physical_total_bytes': total,
            'physical_free_bytes': free_gib * m.GIB,
            'committed_bytes': limit * commit_ppm // 1_000_000,
            'commit_limit_bytes': limit,
            'pagefile_allocated_bytes': pagefile_gib * m.GIB,
        })
    return {'schema': m.SCHEMA, 'samples': rows}


class EnvelopeTests(unittest.TestCase):
    def test_five_samples_stable_never_admit(self):
        out = m.analyze(fixture())
        self.assertEqual(out['state'], 'OBSERVATION_STABLE_ONLY')
        self.assertEqual(out['sample_count'], 5)
        self.assertFalse(out['training_ready'])
        self.assertFalse(out['grants_training_dispatch'])
        self.assertFalse(out['source_or_signer_authenticated'])

    def test_twelve_samples_stable(self):
        out = m.analyze(fixture(n=12))
        self.assertEqual(out['sample_count'], 12)
        self.assertEqual(out['state'], 'OBSERVATION_STABLE_ONLY')

    def test_too_few_samples_fail_closed(self):
        with self.assertRaisesRegex(m.ObservationError, 'SAMPLE_COVERAGE_INVALID'):
            m.analyze(fixture(n=4))

    def test_too_many_samples_fail_closed(self):
        with self.assertRaisesRegex(m.ObservationError, 'SAMPLE_COVERAGE_INVALID'):
            m.analyze(fixture(n=13))

    def test_duplicate_timestamp_fail_closed(self):
        doc = fixture()
        doc['samples'][2]['observed_utc'] = doc['samples'][1]['observed_utc']
        with self.assertRaisesRegex(m.ObservationError, 'INVALID_SAMPLE_SEPARATION'):
            m.analyze(doc)

    def test_too_short_or_long_interval_fail_closed(self):
        for offset in (0.5, 5.1):
            with self.subTest(offset=offset):
                doc = fixture()
                a = datetime.fromisoformat(doc['samples'][0]['observed_utc'])
                doc['samples'][1]['observed_utc'] = (a + timedelta(seconds=offset)).isoformat()
                with self.assertRaisesRegex(m.ObservationError, 'INVALID_SAMPLE_SEPARATION'):
                    m.analyze(doc)

    def test_missing_or_extra_key_fail_closed(self):
        for edit in ('missing', 'extra'):
            with self.subTest(edit=edit):
                doc = fixture()
                if edit == 'missing':
                    del doc['samples'][0]['commit_limit_bytes']
                else:
                    doc['samples'][0]['owner'] = 'not allowed'
                with self.assertRaisesRegex(m.ObservationError, 'INVALID_SAMPLE_FIELDS'):
                    m.analyze(doc)

    def test_bool_negative_and_zero_limit_fail_closed(self):
        for key, value in (('committed_bytes', True), ('physical_free_bytes', -1), ('commit_limit_bytes', 0)):
            with self.subTest(key=key):
                doc = fixture()
                doc['samples'][0][key] = value
                with self.assertRaisesRegex(m.ObservationError, 'INVALID_CAPACITY_VALUE'):
                    m.analyze(doc)

    def test_physical_memory_size_change_fail_closed(self):
        doc = fixture()
        doc['samples'][3]['physical_total_bytes'] += 1
        with self.assertRaisesRegex(m.ObservationError, 'PHYSICAL_MEMORY_IDENTITY_CHANGED'):
            m.analyze(doc)

    def test_low_free_physical_is_hold(self):
        result = m.analyze(fixture(free_gib=1))
        self.assertEqual(result['state'], 'ADVISORY_HOLD')
        self.assertIn('LOW_FREE_PHYSICAL_MEMORY', result['reasons'])

    def test_high_commit_is_hold(self):
        result = m.analyze(fixture(commit_ppm=884000))
        self.assertEqual(result['state'], 'ADVISORY_HOLD')
        self.assertIn('HIGH_COMMIT_FRACTION', result['reasons'])

    def test_caution_boundary_vs_hold(self):
        self.assertEqual(m.analyze(fixture(commit_ppm=800000))['state'], 'OBSERVATION_STABLE_ONLY')
        self.assertEqual(m.analyze(fixture(commit_ppm=850000))['state'], 'ADVISORY_CAUTION')
        self.assertEqual(m.analyze(fixture(free_gib=3))['state'], 'ADVISORY_CAUTION')

    def test_pagefile_swing_classification(self):
        doc = fixture()
        doc['samples'][-1]['pagefile_allocated_bytes'] = 14 * m.GIB
        out = m.analyze(doc)
        self.assertEqual(out['state'], 'ADVISORY_CAUTION')
        self.assertIn('PAGEFILE_ALLOCATION_CHANGE', out['reasons'])
        doc['samples'][-1]['pagefile_allocated_bytes'] = 16 * m.GIB
        out = m.analyze(doc)
        self.assertEqual(out['state'], 'ADVISORY_HOLD')
        self.assertIn('VOLATILE_PAGEFILE_ALLOCATION', out['reasons'])

    def test_pagefile_zero_stays_caution_or_hold(self):
        doc = fixture(pagefile_gib=0)
        self.assertEqual(m.analyze(doc)['state'], 'ADVISORY_CAUTION')
        doc['samples'][4]['pagefile_allocated_bytes'] = 1
        self.assertEqual(m.analyze(doc)['state'], 'ADVISORY_HOLD')

    def test_duplicate_json_keys_rejected(self):
        raw = b'{"schema":"szl.windows-memory-observation/v1","schema":"ignored","samples":[]}'
        with self.assertRaisesRegex(m.ObservationError, 'DUPLICATE_JSON_KEY'):
            m.strict_document(raw)

    def test_noninteger_and_oversize_rejected(self):
        with self.assertRaisesRegex(m.ObservationError, 'NONINTEGER_NUMBER'):
            m.strict_document(b'{"x":1.1}')
        with self.assertRaisesRegex(m.ObservationError, 'INPUT_SIZE_INVALID'):
            m.strict_document(b' ' * (m.MAX_INPUT_BYTES + 1))

    def test_cli_never_changes_control_plane(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'synthetic.json'
            path.write_text(json.dumps(fixture()), encoding='utf-8')
            with patch('sys.stdout') as ignored:
                rc = m.main(['--input', str(path)])
            self.assertEqual(rc, 0)
            self.assertTrue(path.is_file())
            self.assertTrue(ignored.write.called)


if __name__ == '__main__':
    unittest.main()
