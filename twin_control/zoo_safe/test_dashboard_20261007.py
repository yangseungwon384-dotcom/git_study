"""2026-10-07 대시보드 개편의 저장·확인·보관·후보 확정 테스트.

임시 DB와 모의 응답만 사용한다. 운영 데이터와 실제 영상은 건드리지 않는다.
"""
from argparse import Namespace
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from dashboard_policy import DemoGate, ObjectWatch, zone_relation
from dashboard_store import evidence_status
from zoo_vision_jev_new import Database, JevWorker


GATE = DemoGate()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Database(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def event(self, name='giraffe', kind='ZONE_EXIT_DANGER'):
        frame = np.zeros((8, 8, 3), np.uint8)
        return self.db.add_event('CAM', 7, name, 0.8, kind, frame, severity='경고')

    def test_migration_keeps_old_rows_and_can_repeat(self):
        old = self.root / 'old'
        old.mkdir()
        path = old / 'events.sqlite3'
        con = sqlite3.connect(path)
        con.execute('CREATE TABLE events(id TEXT PRIMARY KEY, day TEXT, camera TEXT, active INTEGER, '
                    'row_json TEXT, ended_at TEXT, end_reason TEXT)')
        con.execute('CREATE TABLE decisions(id TEXT PRIMARY KEY, event_id TEXT, requested_at TEXT, status TEXT, '
                    'state_json TEXT, result_json TEXT, error TEXT, model_requested TEXT, question_version TEXT, '
                    'questions_json TEXT)')
        con.execute('CREATE TABLE roi(camera TEXT PRIMARY KEY, json TEXT)')
        con.execute("INSERT INTO events VALUES('e1','2026-10-07','CAM',0,'{\"object\":\"giraffe\"}',NULL,NULL)")
        con.execute("INSERT INTO decisions VALUES('d1','e1','2026-10-07T00:00:00','disabled','{}',NULL,NULL,'x','1.0','{}')")
        con.execute("INSERT INTO roi VALUES('CAM','{\"polygon\":[[0,0],[1,0],[0,1]]}')")
        con.commit()
        con.close()
        db = Database(old)
        self.assertEqual(db.record('e1')['object'], 'giraffe')
        self.assertEqual(len(db.decisions_for('e1')), 1)
        with db.connect() as con:
            roi = con.execute('SELECT json FROM roi WHERE camera=?', ('CAM',)).fetchone()[0]
        self.assertIn('polygon', roi)
        Database(old)
        self.assertTrue((old / 'events.backup-before-v2.sqlite3').is_file())

    def test_ack_is_idempotent_and_does_not_close_the_event(self):
        eid = self.event()
        before = self.db.record(eid)
        first = self.db.acknowledge([eid, eid], '운영')
        second = self.db.acknowledge([eid], '운영')
        after = self.db.record(eid)
        self.assertEqual(first['changed'], [eid])
        self.assertEqual(second['unchanged'], [eid])
        self.assertEqual(after['active'], before['active'])
        self.assertEqual(after['end_reason'], before['end_reason'])
        self.assertEqual(self.db.unacked_count(), 0)
        self.db.unacknowledge([eid])
        self.assertIsNone(self.db.record(eid)['acknowledged_at'])
        self.assertEqual(self.db.unacked_count(), 1)
        self.assertTrue(any(item['action'] == 'unacknowledge' for item in self.db.actions(eid)))

    def test_archive_skips_open_or_unacked_and_ignores_new_ids(self):
        open_id = self.event()
        closed = self.event()
        self.db.end(closed, 'safe_return')
        self.db.acknowledge([closed], '')
        preview = [closed]
        newer = self.event()
        self.db.end(newer, 'safe_return')
        self.db.acknowledge([newer], '')
        ready, skipped = self.db.eligible(preview, 'log')
        self.assertEqual(ready, [closed])
        self.assertIn(open_id, self.db.eligible([open_id, closed], 'log')[1])
        result = self.db.archive(preview, 'log')
        self.assertEqual(result['archived'], [closed])
        visible_ids = {item['id'] for item in self.db.list_records(ack='all', archived='visible')}
        self.assertNotIn(closed, visible_ids)
        self.assertIn(newer, visible_ids)
        self.db.restore([closed], 'log')
        visible_ids = {item['id'] for item in self.db.list_records(ack='all', archived='visible')}
        self.assertIn(closed, visible_ids)

    def test_log_and_snapshot_archive_are_independent(self):
        eid = self.event()
        self.db.end(eid, 'track_lost_not_confirmed_safe')
        self.db.acknowledge([eid], '표시')
        path = Path(self.db.record(eid)['snapshot_path'])
        self.assertTrue(path.is_file())
        self.db.archive([eid], 'log')
        item = self.db.record(eid)
        self.assertIsNotNone(item['log_archived_at'])
        self.assertIsNone(item['snapshot_archived_at'])
        self.assertEqual(item['ack_name'], '미인증:표시')
        self.assertTrue(path.is_file())
        self.db.archive([eid], 'snapshot')
        item = self.db.record(eid)
        self.assertIsNotNone(item['snapshot_archived_at'])
        self.assertTrue(path.is_file())
        gallery = self.db.list_records(ack='all', archived='gallery')
        self.assertNotIn(eid, {row['id'] for row in gallery})
        self.assertEqual(evidence_status(str(path))[0], '')
        self.assertEqual(evidence_status(str(path.with_name('missing.jpg')))[0], '증거 이미지 없음')
        self.assertEqual(evidence_status('')[0], '증거 이미지 없음')
        self.db.restore([eid], 'snapshot')
        self.assertIsNone(self.db.record(eid)['snapshot_archived_at'])
        self.assertIsNotNone(self.db.record(eid)['log_archived_at'])

    def test_review_does_not_create_an_exclude_zone(self):
        eid = self.event()
        self.assertTrue(self.db.review(eid, '오탐', '나무'))
        self.assertEqual(self.db.record(eid)['review_label'], '오탐')
        with self.db.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM roi').fetchone()[0], 0)

    def test_disabled_ai_does_not_call_transport(self):
        holder = {}

        class Engine:
            args = Namespace(jev=False, jev_model='tev1:0.8b', score=False)
            db = self.db

            def finish_job(self, job, status):
                holder['status'] = status

        def transport(*_args):
            raise AssertionError('external call')

        JevWorker(Engine(), transport=transport).submit(
            {'id': 'job', 'event': self.event(), 'state': {'source': {}}})
        self.assertEqual(holder['status'], 'disabled')

    def test_gate_save_is_per_camera(self):
        version = self.db.save_gate('CAM-A', 'giraffe', GATE)
        self.assertEqual(version, 1)
        found = self.db.load_gates(GATE)
        self.assertIn(('CAM-A', 'giraffe'), found)
        self.assertNotIn(('CAM-B', 'giraffe'), found)


class PolicyTests(unittest.TestCase):
    def watch(self):
        return ObjectWatch()

    def step(self, watch, mono, relation, confidence=0.8, frame=None, name='giraffe'):
        return watch.update(mono, frame if frame is not None else mono, confidence, relation,
                            GATE, True, name)

    def test_single_low_confidence_candidate_does_not_open(self):
        watch = self.watch()
        outcome = self.step(watch, 0, 'outside', confidence=0.12, frame=1)
        self.assertNotEqual(outcome['action'], 'open')
        self.assertFalse(watch.event_open)

    def test_confirmed_exit_alerts_and_first_outside_does_not(self):
        watch = self.watch()
        self.step(watch, 0, 'inside', frame=1)
        self.step(watch, 1.0, 'inside', frame=2)
        self.assertTrue(watch.confirmed)
        opened = self.step(watch, 1.2, 'outside', frame=3)
        self.assertEqual(opened['kind'], 'transition')
        fresh = self.watch()
        self.assertNotEqual(self.step(fresh, 0, 'outside', frame=1)['action'], 'open')
        self.step(fresh, 1.0, 'outside', frame=2)
        self.assertEqual(fresh.event_open, True)
        self.assertNotEqual(fresh.last_clear, None)

    def test_boundary_flicker_does_not_duplicate_and_return_reopens(self):
        watch = self.watch()
        self.step(watch, 0, 'inside', frame=1)
        self.step(watch, 1.0, 'inside', frame=2)
        self.step(watch, 1.2, 'outside', frame=3)
        self.assertTrue(watch.event_open)
        for index, mono in enumerate((1.4, 1.6, 1.8)):
            relation = 'boundary' if index % 2 == 0 else 'inside'
            outcome = self.step(watch, mono, relation, frame=4 + index)
            self.assertNotEqual(outcome['action'], 'open')
        self.assertTrue(watch.event_open)
        self.step(watch, 2.0, 'inside', frame=8)
        closed = self.step(watch, 2.9, 'inside', frame=9)
        self.assertEqual(closed['action'], 'close')
        again = self.step(watch, 3.1, 'outside', frame=10)
        self.assertEqual(again['kind'], 'transition')

    def test_same_frame_does_not_add_a_sample(self):
        watch = self.watch()
        self.step(watch, 0, 'inside', frame=1)
        self.step(watch, 0, 'outside', frame=1)
        self.assertEqual(len(watch.samples), 1)

    def test_boundary_margin_uses_short_side_fraction(self):
        self.assertEqual(zone_relation(5, 10, False, False), 'boundary')
        self.assertEqual(zone_relation(20, 10, False, False), 'inside')
        self.assertEqual(zone_relation(-20, 10, False, False), 'outside')
        self.assertEqual(zone_relation(20, 10, True, True), 'bottom_edge')


if __name__ == '__main__':
    unittest.main()
