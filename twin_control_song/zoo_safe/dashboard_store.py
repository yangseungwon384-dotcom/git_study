"""사건 확인, 보관, 탐지 기준의 DB 확장.

열린 WAL 파일을 복사하지 않고 SQLite backup API로 마이그레이션 전 백업을 만든다.
로그 보관과 스냅샷 보관은 서로 다른 시각 컬럼이다. 파일을 지우지 않는다.
"""
from pathlib import Path
import json
import sqlite3
import uuid

from dashboard_policy import DemoGate, gate_from_values

SCHEMA_VERSION = 2
_NEW_COLUMNS = (
    'acknowledged_at', 'ack_name', 'review_label', 'review_note', 'reviewed_at',
    'log_archived_at', 'snapshot_archived_at',
)


def bind(cls):
    cls.migrate = migrate
    cls.list_records = list_records
    cls.record = record
    cls.acknowledge = acknowledge
    cls.unacknowledge = unacknowledge
    cls.review = review
    cls.unacked_count = unacked_count
    cls.occurrence_count = occurrence_count
    cls.eligible = eligible
    cls.archive = archive
    cls.restore = restore
    cls.actions = actions
    cls.load_gates = load_gates
    cls.save_gate = save_gate
    cls.decisions_for = decisions_for


def migrate(self):
    """없는 컬럼과 이력 표만 추가한다. 이미 적용됐으면 아무것도 바꾸지 않는다."""
    with self.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT)')
        current = db.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]
        if current == SCHEMA_VERSION:
            return
        backup = self.root / f'events.backup-before-v{SCHEMA_VERSION}.sqlite3'
        if not backup.exists():
            _backup(db, backup)
        have = {row[1] for row in db.execute('PRAGMA table_info(events)')}
        for name in _NEW_COLUMNS:
            if name not in have:
                db.execute(f'ALTER TABLE events ADD COLUMN {name} TEXT')
        db.execute('''CREATE TABLE IF NOT EXISTS event_actions(
            id TEXT PRIMARY KEY, event_id TEXT, action TEXT, at TEXT, detail TEXT)''')
        db.execute('''CREATE TABLE IF NOT EXISTS detector_settings(
            camera TEXT, class_name TEXT, json TEXT, version INTEGER, changed_at TEXT,
            PRIMARY KEY(camera, class_name))''')
        db.execute('''CREATE TABLE IF NOT EXISTS setting_history(
            id TEXT PRIMARY KEY, camera TEXT, class_name TEXT, before_json TEXT,
            after_json TEXT, changed_at TEXT)''')
        db.execute('CREATE INDEX IF NOT EXISTS events_day_camera ON events(day, camera)')
        db.execute('CREATE INDEX IF NOT EXISTS events_ack ON events(acknowledged_at)')
        db.execute('INSERT INTO schema_migrations(version, applied_at) VALUES(?,?)',
                   (SCHEMA_VERSION, _now(self)))


def list_records(self, day_from=None, day_to=None, camera=None, severity=None,
                 kind=None, ack='all', archived='visible', limit=100, offset=0):
    """화면용 사건 목록. events()의 row_json 계약은 바꾸지 않는다."""
    where, values = _filters(day_from, day_to, camera, severity, kind, ack, archived)
    sql = ('SELECT id, day, camera, active, ended_at, end_reason, row_json, acknowledged_at, '
           'ack_name, review_label, review_note, reviewed_at, log_archived_at, snapshot_archived_at '
           f'FROM events{where} ORDER BY day, rowid LIMIT ? OFFSET ?')
    with self.connect() as db:
        rows = db.execute(sql, (*values, int(limit), int(offset))).fetchall()
    return [_view(row) for row in rows]


def record(self, event_id):
    with self.connect() as db:
        row = db.execute(
            'SELECT id, day, camera, active, ended_at, end_reason, row_json, acknowledged_at, '
            'ack_name, review_label, review_note, reviewed_at, log_archived_at, snapshot_archived_at '
            'FROM events WHERE id=?', (event_id,)).fetchone()
    return _view(row) if row else None


def acknowledge(self, event_ids, display_name=''):
    """이미 확인된 사건은 다시 세지 않는다. 진행 상태와 종료 사유는 바꾸지 않는다."""
    label = '미인증:' + display_name.strip() if str(display_name or '').strip() else '미인증'
    changed, unchanged, missing = [], [], []
    with self.connect() as db:
        for eid in list(event_ids or []):
            row = db.execute(
                'SELECT active, end_reason, acknowledged_at FROM events WHERE id=?', (eid,)).fetchone()
            if row is None:
                missing.append(eid)
                continue
            if row['acknowledged_at']:
                unchanged.append(eid)
                continue
            at = _now(self)
            cur = db.execute(
                'UPDATE events SET acknowledged_at=?, ack_name=? WHERE id=? AND acknowledged_at IS NULL',
                (at, label, eid))
            if cur.rowcount != 1:
                unchanged.append(eid)
                continue
            _action(db, eid, 'acknowledge', at, label)
            after = db.execute('SELECT active, end_reason FROM events WHERE id=?', (eid,)).fetchone()
            if after['active'] != row['active'] or after['end_reason'] != row['end_reason']:
                raise RuntimeError('acknowledge changed incident state')
            changed.append(eid)
    return {'changed': changed, 'unchanged': unchanged, 'missing': missing}


def unacknowledge(self, event_ids):
    changed, unchanged, missing = [], [], []
    with self.connect() as db:
        for eid in list(event_ids or []):
            row = db.execute('SELECT acknowledged_at FROM events WHERE id=?', (eid,)).fetchone()
            if row is None:
                missing.append(eid)
                continue
            if not row['acknowledged_at']:
                unchanged.append(eid)
                continue
            at = _now(self)
            db.execute('UPDATE events SET acknowledged_at=NULL, ack_name=NULL WHERE id=?', (eid,))
            _action(db, eid, 'unacknowledge', at, '')
            changed.append(eid)
    return {'changed': changed, 'unchanged': unchanged, 'missing': missing}


def review(self, event_id, label, note):
    """오탐 표시는 검토 기록이다. 제외 구역이나 학습으로 이어지지 않는다."""
    allowed = {'실제 위험', '오탐', '판단 보류'}
    if label not in allowed:
        raise ValueError('검토 구분을 확인하세요.')
    at = _now(self)
    with self.connect() as db:
        row = db.execute('SELECT id FROM events WHERE id=?', (event_id,)).fetchone()
        if row is None:
            return False
        db.execute('UPDATE events SET review_label=?, review_note=?, reviewed_at=? WHERE id=?',
                   (label, note or '', at, event_id))
        _action(db, event_id, 'review', at, label)
    return True


def unacked_count(self):
    with self.connect() as db:
        return db.execute('SELECT count(*) FROM events WHERE acknowledged_at IS NULL').fetchone()[0]


def occurrence_count(self):
    """보관 여부를 포함한 실제 발생 건수."""
    with self.connect() as db:
        return db.execute('SELECT count(*) FROM events').fetchone()[0]


def eligible(self, event_ids, kind):
    """미리보기에 들어 있던 ID만 다시 검사한다. 목록에 없는 새 사건은 넣지 않는다."""
    column = 'log_archived_at' if kind == 'log' else 'snapshot_archived_at'
    ready, skipped = [], []
    with self.connect() as db:
        for eid in list(event_ids or []):
            row = db.execute(
                f'SELECT active, acknowledged_at, {column} AS archived FROM events WHERE id=?',
                (eid,)).fetchone()
            if row is None or row['active'] or not row['acknowledged_at'] or row['archived']:
                skipped.append(eid)
            else:
                ready.append(eid)
    return ready, skipped


def archive(self, event_ids, kind):
    column = 'log_archived_at' if kind == 'log' else 'snapshot_archived_at'
    action = 'archive_log' if kind == 'log' else 'archive_snapshot'
    ready, skipped = [], []
    with self.connect() as db:
        for eid in list(event_ids or []):
            row = db.execute(
                f'SELECT active, acknowledged_at, {column} AS archived, snapshot_archived_at, '
                'log_archived_at, review_label, acknowledged_at AS ack FROM events WHERE id=?',
                (eid,)).fetchone()
            if row is None or row['active'] or not row['ack'] or row['archived']:
                skipped.append(eid)
                continue
            at = _now(self)
            cur = db.execute(
                f'UPDATE events SET {column}=? WHERE id=? AND active=0 AND acknowledged_at IS NOT NULL '
                f'AND {column} IS NULL', (at, eid))
            if cur.rowcount != 1:
                skipped.append(eid)
                continue
            _action(db, eid, action, at, '')
            ready.append(eid)
    return {'archived': ready, 'skipped': skipped}


def restore(self, event_ids, kind):
    column = 'log_archived_at' if kind == 'log' else 'snapshot_archived_at'
    action = 'restore_log' if kind == 'log' else 'restore_snapshot'
    changed, skipped = [], []
    with self.connect() as db:
        for eid in list(event_ids or []):
            row = db.execute(f'SELECT {column} AS archived FROM events WHERE id=?', (eid,)).fetchone()
            if row is None or not row['archived']:
                skipped.append(eid)
                continue
            at = _now(self)
            db.execute(f'UPDATE events SET {column}=NULL WHERE id=?', (eid,))
            _action(db, eid, action, at, '')
            changed.append(eid)
    return {'restored': changed, 'skipped': skipped}


def actions(self, event_id):
    with self.connect() as db:
        rows = db.execute(
            'SELECT action, at, detail FROM event_actions WHERE event_id=? ORDER BY rowid',
            (event_id,)).fetchall()
    return [dict(row) for row in rows]


def load_gates(self, fallback):
    found = {}
    with self.connect() as db:
        rows = db.execute('SELECT camera, class_name, json, version, changed_at FROM detector_settings').fetchall()
    for row in rows:
        found[(row['camera'], row['class_name'])] = gate_from_values(json.loads(row['json']), fallback)
    return found


def save_gate(self, camera, class_name, gate, at=None):
    """한 카메라·클래스만 저장한다. 다른 카메라에는 쓰지 않는다."""
    at = at or _now(self)
    payload = json.dumps(gate.as_dict(), ensure_ascii=False)
    with self.connect() as db:
        current = db.execute(
            'SELECT json, version FROM detector_settings WHERE camera=? AND class_name=?',
            (camera, class_name)).fetchone()
        before = current['json'] if current else None
        version = (current['version'] if current else 0) + 1
        db.execute(
            'INSERT INTO detector_settings(camera, class_name, json, version, changed_at) VALUES(?,?,?,?,?) '
            'ON CONFLICT(camera, class_name) DO UPDATE SET json=excluded.json, version=excluded.version, '
            'changed_at=excluded.changed_at',
            (camera, class_name, payload, version, at))
        db.execute(
            'INSERT INTO setting_history(id, camera, class_name, before_json, after_json, changed_at) '
            'VALUES(?,?,?,?,?,?)',
            (uuid.uuid4().hex, camera, class_name, before, payload, at))
    return version


def decisions_for(self, event_id):
    with self.connect() as db:
        rows = db.execute(
            'SELECT * FROM decisions WHERE event_id=? ORDER BY rowid DESC', (event_id,)).fetchall()
    return [dict(row) for row in rows]


def evidence_status(path):
    """없거나 깨진 경로는 안내 문장만 돌려준다."""
    if not path:
        return '증거 이미지 없음', None
    file = Path(path)
    if not file.is_file():
        return '증거 이미지 없음', None
    return '', str(file)


def _filters(day_from, day_to, camera, severity, kind, ack, archived):
    clauses, values = [], []
    if day_from:
        clauses.append('day>=?')
        values.append(day_from)
    if day_to:
        clauses.append('day<=?')
        values.append(day_to)
    if camera and camera != '전체':
        clauses.append('camera=?')
        values.append(camera)
    if archived == 'visible':
        clauses.append('log_archived_at IS NULL')
    elif archived == 'archived':
        clauses.append('log_archived_at IS NOT NULL')
    if archived == 'gallery':
        clauses.append('snapshot_archived_at IS NULL')
    if ack == 'open':
        clauses.append('acknowledged_at IS NULL')
    elif ack == 'acked':
        clauses.append('acknowledged_at IS NOT NULL')
    if severity and severity != '전체':
        clauses.append("json_extract(row_json, '$.severity')=?")
        values.append(severity)
    if kind and kind != '전체':
        clauses.append("json_extract(row_json, '$.event_type')=?")
        values.append(kind)
    where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
    return where, values


def _view(row):
    body = json.loads(row['row_json'])
    body.update(id=row['id'], active=bool(row['active']), ended_at=row['ended_at'],
                end_reason=row['end_reason'], acknowledged_at=row['acknowledged_at'],
                ack_name=row['ack_name'], review_label=row['review_label'],
                review_note=row['review_note'], reviewed_at=row['reviewed_at'],
                log_archived_at=row['log_archived_at'],
                snapshot_archived_at=row['snapshot_archived_at'])
    return body


def _action(db, event_id, action, at, detail):
    db.execute('INSERT INTO event_actions(id, event_id, action, at, detail) VALUES(?,?,?,?,?)',
               (uuid.uuid4().hex, event_id, action, at, detail))


def _backup(connection, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(path)
    try:
        connection.backup(dest)
    finally:
        dest.close()


def _now(db_obj):
    from zoo_vision_jev_new import now
    return now().isoformat()


def default_gate(args):
    return DemoGate(candidate_min=float(args.conf), min_span_sec=float(args.confirm_sec),
                    max_gap_sec=float(args.max_gap))
