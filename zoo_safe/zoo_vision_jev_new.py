"""SafeZoo 대시보드 개선본 (zoo_vision_jev_new.py).

실행 방식은 세 가지다.
- 기본 실행: 판단용 JSON만 저장하고 API는 호출하지 않는다.
- --jev: TypeSafe API에 사건 요약 JSON을 전송한다.
- --ollama: 로컬 Ollama의 /v1/systemone에 판단을 요청한다.

AI에는 영상을 보내지 않는다. 탐지 결과, 구역 관계, 관측 시간, 사람 동석 여부만 보낸다.
AI 응답은 기본 경고를 해제하거나 설비를 제어하지 않는다.

이 파일에서 고친 점:
- 동물 이탈과 사람 진입을 다른 규칙으로 본다. 사람 진입 경고는 person_inside가 켜진 때만.
- 박스는 분석한 프레임 위에만 그린다. max_gap보다 오래된 박스는 숨긴다.
- API 실패는 마지막 성공 상태와 분리하고, 같은 상황을 제한 횟수만 다시 요청한다.
- 안전 복귀가 확인되면 같은 ID도 다시 즉시 경고할 수 있다.
- 완료된 AI 답은 이력으로 남기되, jev_ttl이 지난 답은 화면에서 '당시 기록'으로 구분한다.
- 분석 스레드가 죽으면 상태 화면에 표시한다.
- 심각도는 사건 종류, 신뢰도, 지속 시간에 따라 경고/주의/참고로 나눈다.
- 안전구역은 카메라 화면의 가장자리를 클릭해 꼭짓점으로 저장한다.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid

if os.name == 'nt':
    # Anaconda에서 OpenMP가 두 번 로드되며 죽지 않게 하는 우회. 운영 환경에서는 깨끗한 환경을 쓰는 편이 낫다.
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
KST = timezone(timedelta(hours=9))  # 기록 시각은 한국 시간
POLICY_VERSION = 'demo-1'           # JEV에 넘기는 정책 문장의 버전
QUESTION_VERSION = '1.0'            # JEV에 묻는 질문 세트의 버전
# JEV가 고를 수 있는 상황 분류. 값은 모델에게 주는 영어 기준문이다.
CHOICES = {
    'NORMAL': 'Evidence is consistent with permitted activity.',
    'AUTHORIZED_ACTIVITY': 'Verified authorization explains the activity.',
    'BOUNDARY_CONCERN': 'Animal observations support a possible allowed-area violation.',
    'PERSON_INTRUSION_CONCERN': 'Person observations support possible entry into an animal area.',
    'INSUFFICIENT_EVIDENCE': 'Unreliable or missing evidence prevents useful classification.',
    'OTHER': 'Evidence supports a situation not covered by other options.',
}
# --score 를 켰을 때 "지금 운영자가 화면을 확인하는 행동"이 얼마나 적절한지 0~3점 기준.
MAX_JEV_RETRIES = 3          # 같은 상황에 대해 API 실패를 다시 시도하는 횟수
BOX_NOTE_SEC = 1.0            # 이 시간보다 오래된 분석 화면에는 '이전 분석 결과'를 붙인다
PERSON_ALERT_MSG = '안전구역 안에서 사람이 관측되었습니다. 담당자 확인이 필요합니다. (진입 확정 아님)'
SCORE_LEVELS = [
    'The action conflicts with policy or could worsen the situation.',
    'The action offers little direct help with this event.',
    'The action could help but needs a missing prerequisite or another action first.',
    'The action directly addresses the event and is supported by policy and context.',
]


def polygon_crosses(points):
    """꼭짓점을 이은 선이 서로 교차하면 True. 이웃한 변은 꼭짓점을 공유하므로 제외한다."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    n = len(points)
    if n < 4:
        return False
    pts = [(float(p[0]), float(p[1])) for p in points]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if len({i, (i + 1) % n, j, (j + 1) % n}) < 4:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
                return True
    return False


def paint_clicks(rgb, points):
    """클릭한 꼭짓점과, 3개 이상이면 그 안쪽을 현재 화면 위에 그린다."""
    import cv2
    import numpy as np
    canvas = np.array(rgb, copy=True)
    if not points:
        return canvas
    pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in points], np.int32)
    if len(pts) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (40, 200, 80))
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        cv2.polylines(canvas, [pts], True, (40, 220, 90), 2)
    elif len(pts) == 2:
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (40, 220, 90), 2)
    for i, (x, y) in enumerate(pts, start=1):
        cv2.circle(canvas, (int(x), int(y)), 8, (255, 220, 40), -1)
        cv2.putText(canvas, str(i), (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def roi_caption(cfg):
    """화면 왼쪽 위 구역 상태."""
    if not cfg or not cfg.get('polygon'):
        return 'ROI UNSET'
    return 'ARMED' if cfg.get('enabled') else 'DISARMED'


def classify_subject(name, inside, at_bottom, *, enabled, alert_on_exit, person_inside):
    """동물 이탈과 사람 진입을 나눈다.

    동물: 감시가 켜져 있고 발이 구역 밖이면 위험. 기린 카메라(alert_on_exit)만 화면 하단도 위험.
    사람: person_inside가 켜져 있고 발이 구역 안일 때만 위험. 구역 밖이나 화면 하단만으로는 경고하지 않는다.
    """
    if not enabled or inside is None:
        return '', inside, False
    if name == 'person':
        danger = bool(person_inside and inside)
    elif alert_on_exit:
        danger = (not inside) or at_bottom
    else:
        danger = not inside
    return ('DANGER' if danger else 'SAFE'), inside, danger


def judge_severity(kind, confidence, span):
    """사건 종류, 탐지 신뢰도, 위험 지속 시간으로 심각도를 나눈다."""
    if kind == 'PERSON_INTRUSION':
        return '경고' if confidence >= 0.4 else '주의'
    if kind == 'ZONE_EXIT_DANGER':
        return '경고' if confidence >= 0.5 and span >= 1.0 else '주의'
    if confidence < 0.3:
        return '참고'
    return '주의'


def situation_signature(person_present, confidence, name):
    """JEV를 다시 부를 상황 키. 사람 동석, 신뢰도 구간, 클래스가 바뀌면 다른 상황이다."""
    return (bool(person_present), bool(confidence < 0.5), name)


CHOICE_KO = {
    'NORMAL': '평상시 활동으로 보입니다',
    'AUTHORIZED_ACTIVITY': '허가된 작업으로 설명됩니다',
    'BOUNDARY_CONCERN': '안전구역을 벗어났을 가능성이 있습니다',
    'PERSON_INTRUSION_CONCERN': '사람이 동물 구역에 들어왔을 가능성이 있습니다',
    'INSUFFICIENT_EVIDENCE': '증거가 부족해 판단할 수 없습니다',
    'OTHER': '목록에 없는 다른 상황입니다',
}
STATUS_KO = {
    'disabled': '전송 안 함',
    'completed': '판단 도착',
    'stale': '늦은 응답',
    'error': '응답 오류',
    'running': '기다리는 중',
    'unavailable': '연결 실패',
}
CLASS_KO = {
    'giraffe': '기린',
    'elephant': '코끼리',
    'person': '사람',
    'bear': '곰',
    'cat': '고양이',
    'dog': '개',
}
RELATION_KO = {
    'inside': '안전구역 안',
    'outside': '안전구역 밖',
    'bottom_edge': '화면 맨 아래',
}


def _jev_pct(value):
    try:
        return f'{round(float(value) * 100)}%'
    except (TypeError, ValueError):
        return None


def jev_brief(state, status, answers):
    """판단 한 건을 화면용 한글 제목과 설명 문장으로 바꾼다."""
    subject = (state or {}).get('subject') or {}
    evidence = (state or {}).get('evidence') or {}
    detected = subject.get('detected_class')
    name = CLASS_KO.get(detected, detected or '대상')
    if subject.get('class_is_proxy'):
        name += '(대체 분류)'
    where = RELATION_KO.get(evidence.get('current_roi_relation'), '위치 불명')
    choice = (answers or {}).get('situation', {}).get('choice')
    if status == 'disabled':
        headline = 'AI에 보내지 않았습니다. 규칙 경고는 그대로입니다.'
    elif status == 'completed':
        headline = CHOICE_KO.get(choice, '판단이 도착했습니다')
    elif status == 'stale':
        headline = '상황이 바뀐 뒤 도착한 응답입니다. 지금 판단으로 쓰지 마세요.'
    elif status == 'error':
        headline = '응답을 받지 못했습니다. 규칙 경고는 그대로입니다.'
    elif status == 'running':
        headline = '판단을 기다리는 중입니다.'
    elif status == 'unavailable':
        headline = '판단 서버에 연결하지 못했습니다. 규칙 경고는 그대로입니다.'
    else:
        headline = STATUS_KO.get(status, status or '상태 없음')
    lines = [f'본 것: {name}', f'위치: {where}']
    if evidence.get('person_detected_in_same_camera'):
        lines.append('같은 화면에 사람도 있습니다.')
    else:
        lines.append('같은 화면에서 사람은 보이지 않습니다.')
    conf = _jev_pct((answers or {}).get('situation', {}).get('confidence'))
    if status == 'completed' and conf:
        lines.append(f'이 분류에 대한 확신은 {conf}입니다.')
    review = (answers or {}).get('needs_immediate_review', {}).get('noul')
    review_pct = _jev_pct(review)
    if review_pct:
        if float(review) >= 0.5:
            lines.append(f'바로 확인하는 편이 낫습니다({review_pct}).')
        else:
            lines.append(f'당장 확인할 근거는 약합니다({review_pct}).')
    score = (answers or {}).get('action_suitability', {}).get('score')
    if score is not None:
        lines.append(f'화면을 지금 확인하는 행동의 적절성은 3점 중 {score}점입니다.')
    return headline, lines, name, where, CHOICE_KO.get(choice, '-')


def reference_label(item, ttl):
    """이력은 남기되, 지금 화면에서 참고할 답과 당시 기록을 구분한다."""
    if item.get('status') != 'completed' or not item.get('event_active'):
        return '당시 기록'
    try:
        asked = datetime.fromisoformat(item['requested_at'])
    except (TypeError, ValueError):
        return '당시 기록'
    if asked.tzinfo is None:
        asked = asked.replace(tzinfo=KST)
    return '현재 참고' if (now() - asked).total_seconds() <= ttl else '당시 기록'


def now():
    """이벤트·요청 기록에 쓰는 현재 시각(한국 시간)."""
    return datetime.now(KST)


def questions(with_score=False):
    """JEV에게 물어볼 질문 묶음. with_score면 행동 적절성 점수도 추가한다."""
    q = {
        'situation': {
            'type': 'choice',
            'instructions': 'Classify the primary situation using only supplied evidence and policy. '
                            'An ROI crossing is not proof of escape. Unknown authorization or barriers '
                            'must not be assumed. Camera species is not verified identity.',
            'criteria': CHOICES,
        },
        'needs_immediate_review': {
            'type': 'noul',
            'instructions': 'Under the supplied policy, should an operator vhttp://127.0.0.1:7863/erify this event now '
                            'rather than wait for routine review? This is notification, not physical intervention.',
            'criteria': {'true': 'Prompt operator verification is warranted.',
                         'false': 'Evidence supports waiting for routine review.'},
        },
    }
    if with_score:
        q['action_suitability'] = {
            'type': 'score',
            'instructions': 'Evaluate candidate_action using the supplied evidence and policy. '
                            'Do not assume missing resources or permissions.',
            'criteria': SCORE_LEVELS,
        }
    return q


def number(value, low, high):
    """응답 숫자가 low~high 안의 실제 수인지 확인한다. True/False는 숫자로 치지 않는다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('invalid numeric response')
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError('out-of-range response')
    return value


def validate_answer(response, with_score):
    """JEV 응답 형식이 질문과 맞는지 검사한다. 틀린 응답은 저장하지 않고 예외로 올린다."""
    answers = response['answers']
    if not isinstance(response.get('model'), str):
        raise ValueError('missing model version')
    for key, q in questions(with_score).items():
        a = answers[key]
        if a.get('type') != q['type']:
            raise ValueError('answer type mismatch')
        if q['type'] == 'noul':
            number(a['noul'], 0, 1)
            continue
        number(a['confidence'], 0, 1)
        expected = set(CHOICES) if q['type'] == 'choice' else {'0','1','2','3'}
        p = a['probabilities']
        if set(p) != expected or abs(sum(number(v,0,1) for v in p.values())-1) > .02:
            raise ValueError('invalid probability distribution')
        if q['type'] == 'choice':
            if a['choice'] not in expected or p[a['choice']] < max(p.values())-1e-6:
                raise ValueError('invalid choice')
        else:
            score = number(a['score'], 0, 3)
            if abs(score-sum(int(k)*v for k,v in p.items())) > .05:
                raise ValueError('score does not match distribution')
    usage = response.get('usage', {})
    for name in ('input_tokens', 'output_tokens'):
        value = usage[name]
        if type(value) is not int or value < 0:
            raise ValueError('invalid token usage')
    return response


def call_jev(state, args):
    """사건 요약 JSON을 JEV에 보내고, 검사에 통과한 응답만 돌려준다.
    --ollama면 이 PC의 Ollama, 아니면 TypeSafe API를 쓴다.
    """
    payload = {'model': args.jev_model, 'state': state, 'questions': questions(args.score)}
    local = getattr(args, 'provider', 'typesafe') == 'ollama'
    endpoint = args.ollama_url.rstrip('/')+'/v1/systemone' if local else 'https://api.typesafe.ai/v1/systemone'
    headers = {'Content-Type': 'application/json'}
    if not local:
        headers['Authorization'] = 'Bearer '+os.environ['TYPESAFE_API_KEY']
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'),
        headers=headers, method='POST')
    # 실패해도 같은 요청을 바로 다시 보내지 않는다. 상태가 바뀌면 다음 요청이 새로 나간다.
    with urllib.request.urlopen(request, timeout=args.jev_timeout) as response:
        raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError('oversized response')
    return validate_answer(json.loads(raw), args.score)


class Database:
    """알림 사건, JEV 요청/응답, 카메라별 ROI를 SQLite에 저장한다."""
    def __init__(self, root):
        # 테이블이 없으면 만들고, 서버가 죽기 전에 열려 있던 사건·요청은 종료 처리한다.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'events.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS events(
                    id TEXT PRIMARY KEY, day TEXT, camera TEXT, active INTEGER,
                    row_json TEXT, ended_at TEXT, end_reason TEXT);
                CREATE INDEX IF NOT EXISTS day_index ON events(day);
                CREATE TABLE IF NOT EXISTS decisions(
                    id TEXT PRIMARY KEY, event_id TEXT, requested_at TEXT, status TEXT,
                    state_json TEXT, result_json TEXT, error TEXT,
                    model_requested TEXT, question_version TEXT, questions_json TEXT);
                CREATE TABLE IF NOT EXISTS roi(camera TEXT PRIMARY KEY, json TEXT);
            ''')
            db.execute("UPDATE events SET active=0, ended_at=?, end_reason='server_restart' WHERE active=1",
                       (now().isoformat(),))
            db.execute("UPDATE decisions SET status='interrupted' WHERE status IN ('pending','running')")
        from dashboard_store import bind
        bind(Database)
        self.migrate()

    @contextmanager
    def connect(self):
        """쓰기 트랜잭션이 끝나면 연결을 닫는 DB 연결."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_event(self, cid, tid, name, confidence, kind, frame, message=None, zone='ROI', severity='주의'):
        """위험 사건 1건을 저장하고 그 순간의 화면을 jpg로 남긴다. 반환값은 사건 id."""
        import cv2
        eid, date = uuid.uuid4().hex, now()
        path = self.root/'snapshots'/date.date().isoformat()/(eid+'.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, image = cv2.imencode('.jpg', frame)
        if not ok:
            raise RuntimeError('snapshot encoding failed')
        image.tofile(str(path))
        row = dict(detected_at=date.strftime('%Y-%m-%d %H:%M:%S'),camera_id=cid,
                   track_id=tid,object=name,zone=zone,event_type=kind,severity=severity,
                   message=message or '구역 규칙 위반 관측: 담당자 확인 필요 (탈출 확정 아님)',
                   snapshot_path=str(path),confidence=round(confidence,3))
        with self.connect() as db:
            db.execute(
                'INSERT INTO events(id, day, camera, active, row_json, ended_at, end_reason) '
                'VALUES(?,?,?,?,?,?,?)',
                (eid, date.date().isoformat(), cid, 1, json.dumps(row, ensure_ascii=False), None, None))
        return eid

    def end(self, eid, reason):
        """진행 중이던 사건을 끝내고, 그 사건에 대한 완료된 JEV 답은 오래된 것으로 표시한다."""
        with self.connect() as db:
            db.execute('UPDATE events SET active=0, ended_at=?, end_reason=? WHERE id=?',
                       (now().isoformat(),reason,eid))
            db.execute("UPDATE decisions SET status='stale' WHERE event_id=? AND status='completed'",(eid,))

    def invalidate(self,eid):
        """상황이 바뀌어 이전에 받은 JEV 답을 더 이상 현재 판단으로 쓰지 않는다."""
        if eid:
            with self.connect() as db:
                db.execute("UPDATE decisions SET status='stale' WHERE event_id=? AND status='completed'",(eid,))

    def events(self, day=None):
        """오늘(또는 지정한 날짜)의 사건 목록. 대시보드 표와 알림함에 쓴다."""
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(
                'SELECT row_json FROM events WHERE day=? ORDER BY rowid',
                (day or now().date().isoformat(),))]

    def active_count(self):
        """아직 끝나지 않은 사건 수."""
        with self.connect() as db:
            return db.execute('SELECT count(*) FROM events WHERE active=1').fetchone()[0]

    def record_request(self, job, args, status):
        """JEV에 보내기 직전의 요청을 기록한다. --jev가 없으면 status는 disabled."""
        with self.connect() as db:
            db.execute(
                'INSERT INTO decisions(id, event_id, requested_at, status, state_json, result_json, '
                'error, model_requested, question_version, questions_json) VALUES(?,?,?,?,?,?,?,?,?,?)', (
                job['id'],job['event'],now().isoformat(),status,
                json.dumps(job['state'],ensure_ascii=False),None,None,args.jev_model,
                QUESTION_VERSION,json.dumps(questions(args.score))))

    def decision(self, jid, status, response=None, error=None):
        """요청 상태를 갱신한다. completed / stale / error 등."""
        with self.connect() as db:
            db.execute('UPDATE decisions SET status=?,result_json=?,error=? WHERE id=?',
                       (status,json.dumps(response) if response else None,error,jid))

    def latest(self):
        """JEV 판단 탭에 보여줄 최근 요청 50건."""
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                'SELECT d.*, e.active AS event_active FROM decisions d LEFT JOIN events e '
                'ON e.id=d.event_id ORDER BY d.rowid DESC LIMIT 50')]


class JevWorker:
    """JEV 호출을 영상 분석과 분리한 백그라운드 스레드.
    분석 루프가 네트워크를 기다리지 않게, 요청을 줄 세워서 하나씩 보낸다.
    """
    def __init__(self, engine, transport=None):
        self.engine = engine
        self.transport = transport or call_jev
        self.pending = OrderedDict()
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self.run,daemon=True,name='jev-worker')
        self.error_until = 0.0

    def submit(self, job):
        """요청을 큐에 넣는다. 같은 사건의 이전 요청은 덮어쓰고, 32건이 넘으면 가장 오래된 것을 버린다."""
        status = 'pending' if self.engine.args.jev else 'disabled'
        self.engine.db.record_request(job,self.engine.args,status)
        if status == 'disabled':
            # API를 끈 실행은 JSON 기록으로 끝난다. 실패 재시도 대상이 아니다.
            self.engine.finish_job(job,'disabled')
            return
        with self.condition:
            old = self.pending.pop(job['event'],None)
            if old:
                self.engine.db.decision(old['id'],'superseded')
            if len(self.pending) >= 32:
                _, old = self.pending.popitem(last=False)
                self.engine.db.decision(old['id'],'queue_full')
            self.pending[job['event']] = job
            self.condition.notify()

    def current(self, job):
        """이 요청이 아직 그 트랙의 현재 위험 상태인지. 아니면 응답을 버려도 된다."""
        if time.monotonic()-job['created'] > self.engine.args.jev_ttl:
            return False
        with self.engine.lock:
            runtime = self.engine.runtime[job['camera']]
            s = runtime['tracks'].get(job['track'])
            return bool(s and s['event']==job['event'] and s['revision']==job['revision']
                        and runtime['session']==job['session'] and s['danger']
                        and time.monotonic()-s['last'] <= self.engine.args.max_gap)

    def run(self):
        """큐에서 요청을 꺼내 JEV를 호출한다. 오류가 나면 30초 동안 새 호출을 멈춘다."""
        while not self.engine.stop.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.engine.stop.is_set(),timeout=.5)
                if self.engine.stop.is_set():
                    return
                if not self.pending:
                    continue
                _, job = self.pending.popitem(last=False)
            if not self.current(job):
                self.engine.db.decision(job['id'],'stale')
                self.engine.finish_job(job,'stale')
                continue
            if time.monotonic() < self.error_until:
                self.engine.db.decision(job['id'],'unavailable',error='API cooldown; base alarm retained')
                self.engine.finish_job(job,'unavailable')
                continue
            self.engine.db.decision(job['id'],'running')
            started = time.monotonic()
            try:
                # 전송 도중 원본 state가 바뀌지 않도록 복사본을 보낸다.
                response = validate_answer(self.transport(deepcopy(job['state']),self.engine.args),
                                           self.engine.args.score)
                response['latency_ms'] = round((time.monotonic()-started)*1000)
                with self.engine.lock:
                    status = 'completed' if self.current(job) else 'stale'
                    self.engine.db.decision(job['id'],status,response)
                self.engine.finish_job(job,status)
            except Exception as exc:
                # API 키나 응답 본문 전체는 로그에 남기지 않고, 오류 종류만 기록한다.
                message = 'HTTP '+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__
                self.engine.db.decision(job['id'],'error',error=message+'; base alarm retained')
                self.engine.finish_job(job,'error')
                self.error_until = time.monotonic()+30


class LinkedAnalyzer:
    """카메라별 영상 읽기, YOLO 탐지, 구역 판정, 알림, 화면 그리기를 맡는다.
    카메라마다 스레드가 영상을 읽고, 분석 스레드 하나가 YOLO를 돌린다.
    """
    def __init__(self, original, args):
        # original은 zoo_vision.py. 카메라 목록·화면 함수를 거기서 가져온다.
        # runtime 한 칸이 카메라 하나의 현재 상태(프레임, ROI, 트랙, 배너)다.
        self.original, self.args = original,args
        self.db = Database(args.data_dir)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.runtime = {}
        for cam in original.CAMERAS:
            with self.db.connect() as db:
                row = db.execute('SELECT json FROM roi WHERE camera=?',(cam['cam_id'],)).fetchone()
            cfg = json.loads(row[0]) if row else None
            self.runtime[cam['cam_id']] = dict(cam=cam,cfg=cfg,tracks={},session=uuid.uuid4().hex,
                roi_version=cfg.get('version',0) if cfg else 0,frame=None,rendered=None,
                native_size=None,seq=0,stamp=0.,observed_at=None,video_sec=0.,
                box_tracker=IoUTracker(),tracker_session=None,error='',analysis_fps=0.,last_analysis=0.,
                alerted=set(),banner_text='',banner_until=0.,show_roi=True)
        from dashboard_store import default_gate
        self.fallback_gate = default_gate(args)
        self.gates = self.db.load_gates(self.fallback_gate)
        self.model = None
        self.worker = JevWorker(self)
        self.threads = []
        self.analyze_thread = None
        self.boot_error = ''
        self.last_ok = 0.0

    def start(self):
        """JEV 스레드, 카메라별 재생 스레드, YOLO 분석 스레드를 켠다."""
        self.worker.thread.start()
        for cid in self.runtime:
            t=threading.Thread(target=self.capture,args=(cid,),daemon=True)
            t.start(); self.threads.append(t)
        t=threading.Thread(target=self.analyze,daemon=True,name='yolo-analyze')
        self.analyze_thread=t
        t.start();self.threads.append(t)

    def reset(self, runtime, reason):
        """영상이 한 바퀴 돌거나 ROI가 바뀌면 트랙과 배너를 비운다. 진행 중 사건은 종료한다."""
        for s in runtime['tracks'].values():
            if s['event']:
                self.db.end(s['event'],reason)
        runtime['tracks'].clear()
        runtime['session']=uuid.uuid4().hex
        runtime['rendered']=None
        runtime['box_tracker'].reset()
        runtime['alerted'].clear()
        runtime['banner_text']=''
        runtime['banner_until']=0.

    def finish_job(self, job, status):
        """요청 시도와 성공을 분리한다. 실패·만료는 횟수만큼만 다시 보낼 수 있게 inflight를 푼다."""
        with self.lock:
            runtime=self.runtime.get(job['camera'])
            track=runtime['tracks'].get(job['track']) if runtime else None
            if not track or track.get('event')!=job['event'] or track.get('revision')!=job['revision']:
                return
            track['inflight']=False
            if status in ('completed','disabled'):
                track['accepted_signature']=job.get('signature')
                track['retries']=0
            elif status in ('error','unavailable','stale'):
                track['retries']=track.get('retries',0)+1

    def capture(self,cid):
        """영상 파일을 실시간처럼 읽는다. 분석이 늦으면 프레임을 건너뛰고, 끝나면 처음으로 돌아간다."""
        import cv2
        r=self.runtime[cid]
        cap=cv2.VideoCapture(str(r['cam']['file']))
        if not cap.isOpened():
            r['error']='영상 파일을 열 수 없습니다.';cap.release();return
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps=fps if math.isfinite(fps) and 1<=fps<=240 else 25
        start,index=time.monotonic(),0
        try:
            while not self.stop.is_set():
                target=int((time.monotonic()-start)*fps)
                if index>target:
                    self.stop.wait(.02);continue
                if target-index>fps*2:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,target);index=target
                while index<target:
                    if not cap.grab():break
                    index+=1
                ok,frame=cap.read();index+=1
                if not ok:
                    if index<=1:
                        r['error']='영상 디코딩 실패';break
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0);start,index=time.monotonic(),0
                    with self.lock:self.reset(r,'video_loop')
                    self.stop.wait(.03);continue
                h,w=frame.shape[:2]
                with self.lock:
                    r['native_size']=(w,h)
                    # DB에 저장된 ROI가 없으면 zoo_vision의 안전구역을 0~1 좌표로 바꿔 쓴다.
                    # 기린 카메라만 alert_on_exit이 켜져 있어, 구역이 있으면 감시가 바로 활성화된다.
                    if r['cfg'] is None:
                        old=r['cam']['safe_zone']
                        polygon=[[float(x)/(w-1),float(y)/(h-1)] for x,y in old] if old is not None else []
                        valid=bool(polygon) and all(0<=v<=1 for p in polygon for v in p)
                        r['cfg']=dict(polygon=polygon if valid else [],
                            enabled=valid and bool(r['cam']['alert_on_exit']),person_inside=False,
                            reviewed=False,version=0)
                if w>self.args.width:
                    frame=cv2.resize(frame,(self.args.width,round(h*self.args.width/w)))
                with self.lock:
                    r.update(frame=frame,stamp=time.monotonic(),observed_at=now().isoformat(),
                             video_sec=cap.get(cv2.CAP_PROP_POS_MSEC)/1000,seq=r['seq']+1)
                self.stop.wait(1/15)
        except Exception as exc:
            r['error']='영상 오류: '+type(exc).__name__
        finally:
            cap.release()

    def analyze(self):
        """카메라 프레임을 돌아가며 YOLO를 돌리고, 탐지 결과를 화면용으로 저장한 뒤 알림을 판단한다."""
        try:
            import torch
            if self.args.device=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 1))
            self.detector()
        except Exception as exc:
            self.boot_error='분석 시작 실패: '+type(exc).__name__
            with self.lock:
                for r in self.runtime.values():
                    r['error']=self.boot_error
            return
        last={}
        while not self.stop.is_set():
            for cid,r in self.runtime.items():
                if self.stop.is_set():break
                with self.lock:
                    # 영상이 3초 넘게 안 들어오면 그 카메라 추적을 끊는다.
                    if time.monotonic()-r['stamp']>self.args.max_gap:
                        if r['tracks']:self.reset(r,'observation_lost')
                        continue
                    # 새 프레임이 없으면 같은 화면을 다시 추론하지 않는다.
                    if r['frame'] is None or last.get(cid)==r['seq']:continue
                    frame=r['frame'].copy();stamp=r['stamp'];session=r['session']
                    cfg=deepcopy(r['cfg']);last[cid]=r['seq']
                    observed_at,video_sec=r['observed_at'],r['video_sec']
                try:
                    detections=self.detect(r,frame,session,cfg)
                    with self.lock:
                        if session!=r['session']:continue
                        # 박스는 이 프레임과 함께 저장한다. 나중 영상 위에 옛 좌표를 겹치지 않는다.
                        finished=time.monotonic()
                        r['rendered']=(finished,session,detections,frame)
                        r['analysis_fps']=1/max(finished-r['last_analysis'],.001) if r['last_analysis'] else 0
                        r['last_analysis']=finished
                        self.last_ok=finished
                        if finished-stamp>self.args.max_gap:
                            r['error']='분석 지연: 화면은 갱신하고 경고 판단은 보류'
                            continue
                        self.observe(cid,detections,frame,stamp,observed_at,video_sec,cfg,r['seq'])
                        r['error']=''
                except Exception as exc:
                    with self.lock:
                        self.reset(r,'analysis_error');r['error']='분석 오류: '+type(exc).__name__
            self.stop.wait(.02)

    def detector(self):
        # 카메라마다 모델을 따로 두면 메모리가 4배가 되고, 기본 track()은 낮은 신뢰도 박스를 버린다.
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(str(self.args.weights))
        return self.model

    def detect(self,r,frame,session,cfg):
        """YOLO 박스를 읽고 ID를 붙인 뒤, 안전구역 안인지 위험인지를 매긴다."""
        import cv2
        import numpy as np
        model=self.detector()
        if r['tracker_session']!=session:
            r['box_tracker'].reset()
            r['tracker_session']=session
        # 이 카메라가 볼 클래스만 남긴다. 기린 카메라는 giraffe, person.
        ids=[i for i,name in model.names.items() if name in r['cam']['watch']]
        if not ids:raise ValueError('No supported watch classes')
        result=model.predict(frame,classes=ids,conf=self.args.conf,imgsz=self.args.imgsz,
                             device=self.args.device,verbose=False)[0]
        raw=[]
        for box in result.boxes if result.boxes is not None else []:
            raw.append(dict(name=model.names[int(box.cls.item())],
                            confidence=float(box.conf.item()),
                            bbox=[int(v) for v in box.xyxy[0].tolist()]))
        h,w=frame.shape[:2];detections=[]
        poly=self.pixel_polygon(cfg,w,h)
        enabled=bool(cfg and cfg.get('enabled') and poly is not None)
        exclude=self.pixel_polygon({'polygon': cfg.get('exclude_polygon')}, w, h) if cfg else None
        exclude_on=bool(cfg and cfg.get('exclude_reviewed') and exclude is not None)
        if exclude_on:
            kept=[]
            for det in raw:
                fx,fy=self.original.zs.anchor_xy(det['bbox'],h)
                if self.original.zs.in_poly(fx,fy,exclude):
                    continue
                kept.append(det)
            raw=kept
        margin=self.original.zs.BOTTOM_MARGIN
        from dashboard_policy import zone_relation
        short=min(w,h)
        for det in r['box_tracker'].update(raw):
            name=det['name']
            fx,fy=self.original.zs.anchor_xy(det['bbox'],h)
            point=(fx/max(w-1,1),fy/max(h-1,1))
            inside=bool(self.original.zs.in_poly(fx,fy,poly)) if enabled else None
            at_bottom=fy>=h-margin
            gate=self.gate_for(r['cam']['cam_id'], name)
            signed=None
            if enabled:
                import cv2
                signed=float(cv2.pointPolygonTest(poly,(float(fx),float(fy)),True))
            relation=zone_relation(
                signed, gate.boundary_norm*short, at_bottom,
                bool(r['cam'].get('alert_on_exit')) and name!='person') if enabled else 'unknown'
            zone,inside,danger=classify_subject(
                name,inside,at_bottom,enabled=enabled,
                alert_on_exit=bool(r['cam'].get('alert_on_exit')),
                person_inside=bool(cfg and cfg.get('person_inside')))
            if relation=='boundary':
                zone,danger='BOUNDARY',False
            detections.append(dict(track_id=det['track_id'],name=name,confidence=det['confidence'],
                                   bbox=det['bbox'],anchor=point,inside=inside,danger=danger,zone=zone,
                                   at_bottom=at_bottom,relation=relation))
        return detections

    def pixel_polygon(self, cfg, w, h):
        """저장된 0~1 좌표 ROI를 현재 프레임의 픽셀 다각형으로 되돌린다."""
        import numpy as np
        if not cfg or len(cfg.get('polygon') or [])<3:
            return None
        return (np.asarray(cfg['polygon'],np.float32)*[w-1,h-1]).astype(np.int32)

    def gate_for(self, cid, name):
        """저장된 카메라·클래스 기준. 없으면 시연용 기본값이다."""
        return self.gates.get((cid, name), self.fallback_gate)

    def observe(self,cid,detections,frame,stamp,observed_at,video_sec,cfg,frame_key=None):
        """락을 잡은 상태에서 호출된다. 경과 시간은 프레임이 찍힌 단조 시각으로 센다.

        후보 한 장만으로 사건을 만들지 않는다. 안전 쪽이 확인된 객체의 명확한 이탈은 바로 연다.
        처음부터 밖에서 보인 후보는 확인 시간이 지난 뒤에만 연다. AI 답은 이 경고를 끄지 않는다.
        """
        from dashboard_policy import ObjectWatch
        r=self.runtime[cid];tracks=r['tracks']
        person_inside=bool(cfg and cfg.get('person_inside'))
        person_present=any(d['name']=='person' for d in detections)
        for d in detections:
            tid=d['track_id'];s=tracks.get(tid)
            if s and (stamp-s['last']>self.args.max_gap or s['name']!=d['name']):
                if s['event']:self.db.end(s['event'],'track_gap_or_class_change')
                del tracks[tid];s=None
            if s is None:
                s=dict(last=stamp,name=d['name'],event=None,revision=0,danger=False,was_safe=False,
                       pending=None,safe_since=None,history=deque(maxlen=64),watch=ObjectWatch(),
                       last_submit=-1e9,observed_signature=None,accepted_signature=None,
                       inflight=False,retries=0,zone='SAFE')
                tracks[tid]=s
            gate=self.gate_for(cid,d['name'])
            outcome=s['watch'].update(stamp,frame_key,d['confidence'],d.get('relation') or 'unknown',
                                      gate,person_inside,d['name'])
            if outcome['action']=='duplicate':
                continue
            s['last']=stamp
            s['history'].append((stamp,d['confidence']))
            s['zone']=d.get('zone') or s['zone']
            if s['watch'].last_clear=='risk':
                if s['pending'] is None:s['pending']=stamp
            else:
                s['pending']=None
            if s['danger']!=d['danger']:
                s['revision']+=1
                s['inflight']=False
                self.db.invalidate(s['event'])
            s['danger']=d['danger']
            if outcome['action']=='open' and s['event'] is None:
                span=stamp-(s['pending'] if s['pending'] is not None else stamp)
                if d['name']=='person':
                    kind,message,zone='PERSON_INTRUSION',PERSON_ALERT_MSG,'안전구역_안'
                elif outcome['kind']=='transition':
                    kind,message,zone='ZONE_EXIT_DANGER',self.original.zs.ALERT_MSG,'철창밖_위험구역'
                    s['was_safe']=True
                else:
                    kind,message,zone='OUTSIDE_OBSERVED',None,'ROI'
                s['event']=self.db.add_event(cid,tid,d['name'],d['confidence'],kind,frame,message,zone,
                                             judge_severity(kind,d['confidence'],span))
                r['alerted'].add(tid)
                r['banner_text']=message or self.original.zs.ALERT_MSG
                r['banner_until']=time.monotonic()+self.original.zs.ALERT_HOLD_SEC
            elif outcome['action']=='close' and s['event']:
                self.db.end(s['event'],'safe_return')
                s['event']=None
                s['was_safe']=True
                s['pending']=None
                r['alerted'].discard(tid)
                s['accepted_signature']=None
                s['inflight']=False
                s['retries']=0
            if s['event'] and s['watch'].last_clear=='risk':
                self.maybe_submit(r,cid,tid,s,d,cfg,observed_at,video_sec,person_present,stamp)
        for tid,s in list(tracks.items()):
            if stamp-s['last']>self.args.max_gap:
                if s['event']:self.db.end(s['event'],'track_lost_not_confirmed_safe')
                if s.get('watch'):s['watch'].mark_closed()
                del tracks[tid]

    def maybe_submit(self,r,cid,tid,s,d,cfg,observed_at,video_sec,person_present,stamp):
        """상황 키가 마지막 성공과 다를 때만 요청한다. 실패하면 횟수 안에서만 다시 보낸다."""
        signature=situation_signature(person_present,d['confidence'],d['name'])
        if signature!=s.get('observed_signature'):
            s['observed_signature']=signature
            s['retries']=0
            s['inflight']=False
            s['revision']+=1
            self.db.invalidate(s['event'])
        if signature==s.get('accepted_signature') or s.get('inflight'):
            return
        if s.get('retries',0)>=MAX_JEV_RETRIES or stamp-s['last_submit']<self.args.cooldown:
            return
        s['inflight']=True
        s['last_submit']=stamp
        s['revision']+=1
        state=self.build_state(cid,d,s,cfg,observed_at,video_sec,person_present,stamp)
        job=dict(id=uuid.uuid4().hex,event=s['event'],camera=cid,track=tid,session=r['session'],
                 revision=s['revision'],created=time.monotonic(),state=state,signature=signature)
        self.worker.submit(job)

    def build_state(self,cid,d,s,cfg,observed_at,video_sec,person_present,stamp):
        """JEV에 보낼 사건 요약. 영상 전체가 아니라 탐지 결과와 구역 관계만 담는다."""
        samples=list(s['history']);gaps=[b[0]-a[0] for a,b in zip(samples,samples[1:])]
        # 표범은 cat/dog/bear, 판다는 bear로 대신 탐지한다. 그 경우 실제 종명이 아니다.
        proxy=('LEOPARD' in cid or 'PANDA' in cid) and d['name']!='person'
        animal_rule='alert when an animal foot is outside the ROI'
        if self.runtime[cid]['cam'].get('alert_on_exit'):
            animal_rule+=' or on the bottom edge'
        animal_rule+='; not proof of escape'
        state=dict(schema_version='1.0',source=dict(camera_id=cid,mode='recorded_video_demo',
            observed_at=observed_at,video_position_sec=round(video_sec,2),roi_version=cfg['version'],
            observation_age_ms=round((time.monotonic()-stamp)*1000)),
            subject=dict(detected_class=d['name'],class_is_proxy=proxy,
                         detector_confidence_min=round(min(c for _,c in samples),3)),
            evidence=dict(current_roi_relation=(
                'bottom_edge' if d.get('at_bottom') and d['inside'] and d['name']!='person'
                else 'inside' if d['inside'] else 'outside'),
                previously_observed_safe=s['was_safe'],violation_observed_span_sec=round(stamp-s['pending'],2),
                recent_violation_samples=len(samples),max_recent_gap_sec=round(max(gaps,default=0),2),
                person_detected_in_same_camera=person_present),
            context=dict(roi_reviewed_by_operator=cfg['reviewed'],authorized_work='unknown',
                         physical_barrier_status='unknown',person_identity='unknown'),
            policy=dict(version=POLICY_VERSION,roi_meaning='permitted animal area',
                animal_outside=animal_rule,
                person_inside='alert only when a person foot is inside the ROI' if cfg.get('person_inside')
                              else 'person inside ROI is not an alert',
                person_outside='not an alert',
                safety_rule='AI must not clear the rule alarm or control equipment. '
                            'Missing information is not evidence of safety.'))
        if self.args.score:
            state['candidate_action']=dict(id='VERIFY_CURRENT_VIEW',
                description='Ask the operator to review this camera and its ROI boundary now.')
        return state

    def step_all(self):
        """Gradio가 0.2초마다 부르는 화면 갱신.
        박스는 분석한 프레임 위에만 그린다. max_gap보다 오래되면 박스를 숨긴다.
        """
        import cv2
        import numpy as np
        out=[]
        with self.lock:
            snapshots=[]
            for r in self.runtime.values():
                rendered=r['rendered']
                shot=None if not rendered or rendered[3] is None else rendered[3].copy()
                snapshots.append((r['frame'],deepcopy(r['cfg']),rendered,r['session'],
                                  r['banner_text'],r['banner_until'],shot,r.get('show_roi', True)))
        for frame,cfg,rendered,session,banner_text,banner_until,shot,show_roi in snapshots:
            if frame is None and shot is None:out.append(None);continue
            age=time.monotonic()-rendered[0] if rendered else None
            fresh=bool(rendered and rendered[1]==session and shot is not None and age<=self.args.max_gap)
            # 박스가 맞으려면 분석한 프레임을 같이 보여 준다. max_gap을 넘으면 박스를 숨기고 최신 영상을 쓴다.
            canvas=(shot if fresh else frame).copy()
            if canvas is None:out.append(None);continue
            h,w=canvas.shape[:2]
            if show_roi and cfg and cfg.get('polygon'):
                pts=(np.asarray(cfg['polygon'])*[w-1,h-1]).astype(np.int32)
                cv2.polylines(canvas,[pts],True,(0,220,230),2)
                cv2.putText(canvas,roi_caption(cfg),(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,230),2)
            if show_roi and cfg and cfg.get('exclude_polygon'):
                extra=(np.asarray(cfg['exclude_polygon'])*[w-1,h-1]).astype(np.int32)
                cv2.polylines(canvas,[extra],True,(0,140,255),2)
                cv2.putText(canvas,'탐지 제외',(12,48),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,140,255),2)
            if fresh:
                for d in rendered[2]:
                    x1,y1,x2,y2=d['bbox'];color=(0,0,255) if d['danger'] else (0,210,0)
                    zone=d.get('zone') or ''
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    cv2.putText(canvas,f"{d['name']} {d['confidence']:.2f} #{d['track_id']} {zone}",(x1,max(18,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                if age>=BOX_NOTE_SEC:
                    cv2.putText(canvas,f'이전 분석 결과 {age:.1f}s',(12,h-16),
                                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,230),2)
            elif frame is not None:
                cv2.putText(canvas,'분석 결과 만료 — 박스 숨김',(12,h-16),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,230),2)
            # zoo_safety.draw_alert_banner: 빨간 띠에 이탈 안내 문장을 올린다.
            if banner_text and time.monotonic()<banner_until:
                self.original.zs.draw_alert_banner(canvas,banner_text)
            out.append(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        return out

    def close(self):
        """서버가 끝날 때 스레드를 멈추고 남은 사건을 종료한다."""
        self.stop.set()
        with self.worker.condition:self.worker.condition.notify_all()
        for thread in self.threads:thread.join(timeout=5)
        if self.worker.thread.is_alive():self.worker.thread.join(timeout=self.args.jev_timeout+1)
        with self.lock:
            for r in self.runtime.values():self.reset(r,'server_shutdown')


def box_iou(a, b):
    """두 박스의 겹침 비율. 추적기가 ID를 안 줄 때 같은 물체인지 판별한다."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union else 0.0


class IoUTracker:
    """YOLO predict 결과에 안정적인 track id를 붙인다.

    Ultralytics 기본 tracktrack은 new_track_thresh가 0.7이라
    기린·판다처럼 신뢰도가 낮은 탐지를 ID 없이 돌려준다.
    """

    def __init__(self, iou_min=0.3, hold=8):
        self.iou_min = iou_min
        self.hold = hold
        self.next_id = 1
        self.tracks = {}

    def reset(self):
        """영상이 반복되거나 ROI가 바뀌면 ID 번호를 처음부터 다시 센다."""
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections):
        """이전 박스와 가장 많이 겹치는 같은 클래스 박스에 기존 ID를 붙인다.
        겹치는 이전 박스가 없으면 새 ID를 주고, 여러 프레임 동안 안 보이면 지운다.
        """
        previous = set(self.tracks)
        pairs = []
        for tid, track in self.tracks.items():
            for index, det in enumerate(detections):
                if track['name'] != det['name']:
                    continue
                score = box_iou(track['bbox'], det['bbox'])
                if score >= self.iou_min:
                    pairs.append((score, tid, index))
        pairs.sort(reverse=True)
        assigned_tracks = set()
        assigned_dets = set()
        out = [None] * len(detections)
        for _score, tid, index in pairs:
            if tid in assigned_tracks or index in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(index)
            self.tracks[tid]['bbox'] = detections[index]['bbox']
            self.tracks[tid]['missed'] = 0
            det = dict(detections[index])
            det['track_id'] = tid
            out[index] = det
        for index, det in enumerate(detections):
            if index in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = dict(bbox=det['bbox'], name=det['name'], missed=0)
            det = dict(det)
            det['track_id'] = tid
            out[index] = det
        for tid in previous:
            if tid in assigned_tracks:
                continue
            self.tracks[tid]['missed'] += 1
            if self.tracks[tid]['missed'] > self.hold:
                del self.tracks[tid]
        return out


def load_original(path):
    """zoo_vision.py를 이 프로세스 안으로 불러온다. 화면 구성과 카메라 목록을 재사용하기 위해서다.
    불러오는 순간 zoo_vision.py가 gradio를 import 하므로, gradio가 있는 환경에서만 실행된다.
    """
    if not (path/'zoo_vision.py').is_file() or not (path/'zoo_safety.py').is_file():
        raise FileNotFoundError('--source-dir must contain zoo_vision.py and zoo_safety.py')
    sys.path.insert(0,str(path))
    spec=importlib.util.spec_from_file_location('safezoo_original',path/'zoo_vision.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_app(original, engine):
    """왼쪽 메뉴와 화면별 관제 기능을 붙인다. 영상 분석은 화면 갱신과 분리되어 있다."""
    from dashboard_ui import build_dashboard
    return build_dashboard(original, engine)


def parse_args():
    """실행 옵션. 신뢰도 기본값은 0.1이라 점수가 낮은 기린·판다 박스도 남긴다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path(r'C:\code\code_test\zoo'))  # 영상·zoo_vision.py 위치
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'zoo_jev_data_new')  # 이 파일 전용 SQLite와 스냅샷
    parser.add_argument('--weights',type=Path)  # 비우면 yolo26n.pt를 찾는다
    parser.add_argument('--port',type=int,default=7863)  # 대시보드 주소 포트
    parser.add_argument('--width',type=int,default=960)  # 이보다 넓은 영상은 줄여서 추론한다
    parser.add_argument('--imgsz',type=int,default=640)  # YOLO 입력 크기
    parser.add_argument('--conf',type=float,default=.1)  # 이 점수 미만 박스는 버린다
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--confirm-sec',type=float,default=1.0)  # 기린 이탈 즉시 알림 외의 위험은 이 시간만큼 유지돼야 기록
    parser.add_argument('--max-gap',type=float,default=3.0)  # 이 시간 넘게 안 보이면 트랙을 놓친 것으로 본다
    parser.add_argument('--cooldown',type=float,default=10.0)  # 같은 사건의 JEV 재전송 최소 간격(초)
    backend=parser.add_mutually_exclusive_group()
    backend.add_argument('--jev',action='store_true',help='Send incident metadata to TypeSafe API')
    backend.add_argument('--ollama',action='store_true',help='Use local Ollama decision model, no API key')
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    parser.add_argument('--score',action='store_true',help='Also evaluate operator review action suitability')
    parser.add_argument('--jev-model',default=None,help='Default: tev1:0.8b for Ollama, jev-1.13.0 for TypeSafe')
    parser.add_argument('--jev-timeout',type=float,default=None)
    parser.add_argument('--jev-ttl',type=float,default=None)
    parser.add_argument('--share',action='store_true')
    args=parser.parse_args()
    args.provider='ollama' if args.ollama else 'typesafe'
    args.jev_model=args.jev_model or ('tev1:0.8b' if args.ollama else 'jev-1.13.0')
    args.jev_timeout=args.jev_timeout if args.jev_timeout is not None else (30 if args.ollama else 5)
    args.jev_ttl=args.jev_ttl if args.jev_ttl is not None else (45 if args.ollama else 10)
    if args.ollama:
        url=urlparse(args.ollama_url)
        if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1','::1') or url.username or url.password or url.query or url.fragment or url.path not in ('','/'):
            parser.error('--ollama-url에는 이 PC의 http://localhost:포트 주소만 사용할 수 있습니다.')
    if args.jev and not os.environ.get('TYPESAFE_API_KEY'):
        parser.error('--jev 실행에는 TYPESAFE_API_KEY 환경변수가 필요합니다.')
    # --ollama도 외부로 판단을 보내는 모드로 취급한다. 둘 다 없으면 JSON만 저장한다.
    args.jev=args.jev or args.ollama
    if args.share and not os.environ.get('SAFEZOO_PASSWORD'):
        parser.error('--share 실행에는 SAFEZOO_PASSWORD 환경변수가 필요합니다. ID는 zoo입니다.')
    if not (160<=args.width<=1920 and 128<=args.imgsz<=1280 and 0<args.conf<1 and
            0<args.confirm_sec<=30 and .1<args.max_gap<=30 and args.cooldown>=1 and
            0<args.jev_timeout<=30 and 0<args.jev_ttl<=60):parser.error('설정 범위 확인 필요')
    if args.weights is None:
        choices=[args.source_dir.parent/'unit5'/'yolo26n.pt',args.source_dir/'yolo26n.pt']
        args.weights=next((p for p in choices if p.is_file()),None)
        if args.weights is None:parser.error('--weights에 로컬 YOLO 모델 경로를 지정하세요.')
    return args


JEV_CSS = """
#jev-panel, #jev-panel .prose, #jev-panel .prose p, #jev-panel .prose li,
#jev-panel .prose h1, #jev-panel .prose h2, #jev-panel .prose h3, #jev-panel .prose strong {
  color: #f4f4f5 !important;
}
#jev-panel .jev-mode { color: #a1a1aa !important; margin: 0 0 8px; }
#jev-panel .jev-card {
  background: #1c1c1c;
  border: 1px solid #3f3f46;
  border-left: 4px solid #86efac;
  border-radius: 12px;
  padding: 16px 18px;
  margin: 0 0 14px;
}
#jev-panel .jev-card h2 {
  color: #fafafa !important;
  font-size: 1.45rem;
  line-height: 1.35;
  margin: 4px 0 10px;
}
#jev-panel .jev-card p, #jev-panel .jev-card li {
  color: #e4e4e7 !important;
  font-size: 1.05rem;
  line-height: 1.55;
}
#jev-panel .jev-kicker { color: #d4d4d8 !important; margin: 0; }
#jev-panel .jev-now { color: #86efac !important; font-weight: 700; }
#jev-panel .jev-old { color: #fde68a !important; font-weight: 700; }
#jev-panel .jev-note { color: #a1a1aa !important; margin: 8px 0 0; }
#jev-panel .jev-error { color: #fca5a5 !important; }
#jev-panel .block, #jev-panel .label-wrap, #jev-panel .empty {
  background: #1c1c1c !important;
}
#jev-panel label, #jev-panel .label-wrap span {
  color: #f4f4f5 !important;
  background: transparent !important;
}
#jev-panel .json-holder {
  background: #141414 !important;
  border: 1px solid #3f3f46 !important;
  border-radius: 8px;
}
#jev-panel .json-node {
  --text-color: #f4f4f5;
  --key-color: #7dd3fc;
  --string-color: #fca5a5;
  --number-color: #86efac;
  --punctuation-color: #e4e4e7;
  --bracket-color: #fde68a;
  --square-bracket-color: #d8b4fe;
  --line-number-color: #a1a1aa;
  --separator-color: #52525b;
}
#jev-panel table, #jev-panel th, #jev-panel td,
#jev-panel [role="gridcell"], #jev-panel [role="columnheader"] {
  background: #1c1c1c !important;
  color: #f4f4f5 !important;
  border-color: #3f3f46 !important;
}
"""


def main():
    """옵션 해석 → 원본 대시보드 로드 → 분석 시작 → Gradio 서버. 종료 시 스레드를 정리한다."""
    args=parse_args();original=load_original(args.source_dir)
    engine=LinkedAnalyzer(original,args)
    demo=build_app(original,engine)
    engine.start()
    print(f'SafeZoo: http://127.0.0.1:{args.port} | decision: {args.provider if args.jev else "JSON only"} / {args.jev_model}')
    try:
        from dashboard_ui import DASHBOARD_CSS
        demo.launch(server_name='127.0.0.1',server_port=args.port,share=args.share,
                    auth=('zoo',os.environ['SAFEZOO_PASSWORD']) if args.share else None,
                    theme=original.build_theme(),css=original.load_css()+DASHBOARD_CSS+JEV_CSS,
                    allowed_paths=[str(engine.db.root/'snapshots')],
                    blocked_paths=[str(engine.db.path)])
    finally:
        engine.close()


if __name__=='__main__':
    main()  # python zoo_vision_jev.py 로 직접 실행할 때만 서버를 연다.
