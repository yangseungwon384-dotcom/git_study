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
    'lion': '사자',
    'panda': '판다',
    'jaguar': '재규어',
    'person': '사람',
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
            'instructions': 'Under the supplied policy, should an operator verify this event now '
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
            db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',
                       (eid,date.date().isoformat(),cid,1,json.dumps(row,ensure_ascii=False),None,None))
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
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?)',(
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
                roi_version=cfg.get('version',0) if cfg else 0,frame=None,preview=None,rendered=None,
                clips=deque(maxlen=96),exit_frames=deque(maxlen=4),play_origin=None,play_cursor=None,fps=24.0,
                track_motion={},pending_follow=None,
                native_size=None,seq=0,stamp=0.,observed_at=None,video_sec=0.,
                box_tracker=IoUTracker(),tracker_session=None,error='',analysis_fps=0.,last_analysis=0.,
                alerted=set(),banner_text='',banner_until=0.)
        self.model = None
        self.worker = JevWorker(self)
        self.threads = []
        self.analyze_thread = None
        self.boot_error = ''
        self.last_ok = 0.0
        self.stream_lock = threading.Lock()
        self.stream_jpegs = []
        self.stream_port = None
        self.stream_httpd = None
        self.timeline = {}
        self.media_files = []

    def start(self):
        """JEV 스레드, 카메라별 재생 스레드, YOLO 분석 스레드를 켠다."""
        self.worker.thread.start()
        for cid in self.runtime:
            t=threading.Thread(target=self.capture,args=(cid,),daemon=True)
            t.start(); self.threads.append(t)
        t=threading.Thread(target=self.analyze,args=('coco',),daemon=True,name='yolo-coco')
        self.analyze_thread=t
        t.start();self.threads.append(t)
        t=threading.Thread(target=self.analyze,args=('yoloe',),daemon=True,name='yolo-yoloe')
        t.start();self.threads.append(t)

    def reset(self, runtime, reason):
        """영상이 한 바퀴 돌거나 ROI가 바뀌면 트랙과 배너를 비운다. 진행 중 사건은 종료한다."""
        for s in runtime['tracks'].values():
            if s['event']:
                self.db.end(s['event'],reason)
        runtime['tracks'].clear()
        runtime['session']=uuid.uuid4().hex
        runtime['rendered']=None
        runtime['clips'].clear()
        runtime['play_origin']=None
        runtime['play_cursor']=None
        runtime['track_motion'].clear()
        runtime['pending_follow']=None
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
        """영상 파일을 원본 FPS로 읽는다. 프레임을 건너뛰지 않고, 끝나면 처음으로 돌아간다."""
        import cv2
        r=self.runtime[cid]
        cap=cv2.VideoCapture(str(r['cam']['file']))
        if not cap.isOpened():
            r['error']='영상 파일을 열 수 없습니다.';cap.release();return
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps=fps if math.isfinite(fps) and 1<=fps<=240 else 25.0
        interval=1.0/fps
        next_at=time.monotonic()
        index=0
        try:
            while not self.stop.is_set():
                ok,frame=cap.read()
                index+=1
                if not ok:
                    if index<=1:
                        r['error']='영상 디코딩 실패';break
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0);index=0
                    next_at=time.monotonic()
                    with self.lock:self.reset(r,'video_loop')
                    self.stop.wait(.03);continue
                h,w=frame.shape[:2]
                with self.lock:
                    r['native_size']=(w,h)
                    # DB에 저장된 ROI가 없으면 카메라의 안전구역을 0~1 좌표로 바꿔 쓴다.
                    if r['cfg'] is None:
                        old=r['cam']['safe_zone']
                        polygon=[[float(x)/(w-1),float(y)/(h-1)] for x,y in old] if old is not None else []
                        valid=bool(polygon) and all(0<=v<=1 for p in polygon for v in p)
                        r['cfg']=dict(polygon=polygon if valid else [],
                            enabled=valid,person_inside=False,
                            reviewed=False,version=0)
                preview=frame
                if w>640:
                    preview=cv2.resize(frame,(640,round(h*640/w)))
                with self.lock:
                    seq=r['seq']+1
                    r['fps']=fps
                    r['clips'].append((seq,preview))
                    if r['play_origin'] is None:
                        r['play_origin']=(time.monotonic(),seq)
                    r.update(frame=frame,preview=preview,stamp=time.monotonic(),observed_at=now().isoformat(),
                             video_sec=cap.get(cv2.CAP_PROP_POS_MSEC)/1000,seq=seq)
                    if r['cam'].get('alert_on_exit'):
                        sec = r['video_sec']
                        hist = r['exit_frames']
                        if not hist or sec < hist[-1][0] - 0.05 or sec - hist[-1][0] >= 0.35:
                            hist.append((sec, frame.copy()))
                next_at+=interval
                delay=next_at-time.monotonic()
                if delay>0:
                    if self.stop.wait(delay):
                        break
                else:
                    next_at=time.monotonic()
        except Exception as exc:
            r['error']='영상 오류: '+type(exc).__name__
        finally:
            cap.release()

    def analyze(self, kind='coco'):
        """같은 종류의 카메라만 돌아가며 탐지한다. COCO와 열린 어휘는 동시에 돈다."""
        try:
            import torch
            torch.set_num_threads(2)
            self.model_for(kind)
        except Exception as exc:
            message='분석 시작 실패: '+type(exc).__name__
            with self.lock:
                self.boot_error=message
                for r in self.runtime.values():
                    if (r['cam'].get('kind') or 'coco')==kind:
                        r['error']=message
            return
        last={}
        while not self.stop.is_set():
            for cid,r in list(self.runtime.items()):
                if self.stop.is_set():break
                if (r['cam'].get('kind') or 'coco')!=kind:
                    continue
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
                    exit_frames=[(sec, image.copy()) for sec, image in r.get('exit_frames', ())]
                try:
                    detections, remembered = self.detect(
                        r, frame, session, cfg, exit_frames, video_sec)
                    with self.lock:
                        if session!=r['session']:continue
                        finished=time.monotonic()
                        r['rendered']=(finished,session,detections,last[cid])
                        motion=r['track_motion']
                        seen=set()
                        for det in detections:
                            seen.add(det['track_id'])
                            hist=motion.setdefault(det['track_id'],deque(maxlen=2))
                            hist.append((last[cid],det['norm'],det['track_id'],det['name'],
                                         det['confidence'],det['danger'],det.get('zone') or ''))
                        for tid in list(motion):
                            if tid not in seen and motion[tid] and last[cid]-motion[tid][-1][0]>48:
                                del motion[tid]
                        r['pending_follow']=(session, last[cid], [
                            (det['norm'], det['track_id'], det['name'], det['confidence'])
                            for det in detections])
                        self.note_timeline(cid, video_sec, detections)
                        for sec, extra in remembered:
                            self.note_timeline(cid, sec, extra)
                        r['analysis_fps']=1/max(finished-r['last_analysis'],.001) if r['last_analysis'] else 0
                        r['last_analysis']=finished
                        self.last_ok=finished
                        self.observe(cid,detections,frame,stamp,observed_at,video_sec,cfg)
                        r['error']=''
                except Exception as exc:
                    with self.lock:
                        self.reset(r,'analysis_error');r['error']='분석 오류: '+type(exc).__name__
            self.stop.wait(.01)

    def model_for(self, kind):
        """COCO 모델과 열린 어휘 모델을 나눠 쓴다. 카메라마다 가중치를 다시 읽지 않는다."""
        if not hasattr(self, '_model_lock'):
            self._model_lock = threading.Lock()
        with self._model_lock:
            if not hasattr(self, 'models'):
                self.models = {}
                self.yoloe_prompt = None
            if kind not in self.models:
                if kind == 'yoloe':
                    from ultralytics import YOLOE
                    root = share_root()
                    folders = (root, Path(__file__).resolve().parent)
                    choices = [folder / 'yoloe-26n-seg.pt' for folder in folders]
                    path = next((item for item in choices if item.is_file()), None)
                    if path is None:
                        raise FileNotFoundError(
                            'yoloe-26n-seg.pt를 공유 폴더 안에 두세요: ' + str(root))
                    self.models[kind] = YOLOE(str(path))
                else:
                    from ultralytics import YOLO
                    self.models[kind] = YOLO(str(self.args.weights))
            model = self.models[kind]
        return model

    def fence_exit_marks(self, model, frame, video_sec, history, cfg):
        """울타리 앞 기린만 다시 찾는다.

        전체 화면에서는 멀리 있는 기린만 잡히고, 울타리 밖으로 나온 머리는
        기린이 아니거나 사람으로만 잡히는 구간이 있다. 최근 몇 프레임의
        아래쪽만 잘라 보고, 발이 구역 밖인 박스만 이탈로 남긴다.
        """
        giraffe_ids = [i for i, name in model.names.items() if name == 'giraffe']
        person_ids = [i for i, name in model.names.items() if name == 'person']
        frames = []
        seen = set()
        for sec, image in list(history) + [(video_sec, frame)]:
            try:
                key = round(float(sec or 0), 2)
            except (TypeError, ValueError):
                continue
            if image is None or key in seen:
                continue
            seen.add(key)
            frames.append((key, image))
        frames = frames[-4:]
        if not frames or not giraffe_ids:
            return [], []
        crops, meta = [], []
        images = {}
        for key, image in frames:
            height, width = image.shape[:2]
            y0, x0, x1 = int(height * 0.55), int(width * 0.32), int(width * 0.80)
            crop = image[y0:, x0:x1]
            if crop.size == 0:
                continue
            crops.append(crop)
            meta.append((key, width, height, x0, y0))
            images[key] = image
        if not crops:
            return [], []
        found = {key: [] for key, *_rest in meta}
        predicted = model.predict(crops, classes=giraffe_ids, conf=0.02, imgsz=640,
                                  device=self.args.device, verbose=False)
        for result, (key, width, height, x0, y0) in zip(predicted, meta):
            poly = self.pixel_polygon(cfg, width, height)
            for box in result.boxes if result.boxes is not None else []:
                xy = [int(v) for v in box.xyxy[0].tolist()]
                xyxy = [xy[0] + x0, xy[1] + y0, xy[2] + x0, xy[3] + y0]
                if foot_outside(xyxy, poly, height):
                    found[key].append(dict(name='giraffe', confidence=float(box.conf.item()), bbox=xyxy))
        missed = [item for item in meta if not found[item[0]]]
        if missed and person_ids:
            person_crops, person_meta = [], []
            for key, width, height, _x0, _y0 in missed:
                image = images.get(key)
                if image is None:
                    continue
                py0, px0, px1 = int(height * 0.66), int(width * 0.50), int(width * 0.78)
                crop = image[py0:, px0:px1]
                if crop.size == 0:
                    continue
                person_crops.append(crop)
                person_meta.append((key, width, height, px0, py0))
            if person_crops:
                person_pred = model.predict(
                    person_crops, classes=person_ids, conf=0.02, imgsz=640,
                    device=self.args.device, verbose=False)
                for result, (key, width, height, px0, py0) in zip(person_pred, person_meta):
                    poly = self.pixel_polygon(cfg, width, height)
                    for box in result.boxes if result.boxes is not None else []:
                        xy = [int(v) for v in box.xyxy[0].tolist()]
                        xyxy = [xy[0] + px0, xy[1] + py0, xy[2] + px0, xy[3] + py0]
                        if not fence_head(xyxy, width, height):
                            continue
                        if foot_outside(xyxy, poly, height):
                            found[key].append(dict(
                                name='giraffe', confidence=float(box.conf.item()), bbox=xyxy))
        try:
            current_key = round(float(video_sec or 0), 2)
        except (TypeError, ValueError):
            current_key = None
        current = found.get(current_key, []) if current_key is not None else []
        remembered = []
        for key, width, height, _x0, _y0 in meta:
            boxes = found.get(key) or []
            if not boxes or key == current_key:
                continue
            remembered.append((key, [
                exit_timeline_det(det, width, height) for det in boxes]))
        return current, remembered

    def detect(self, r, frame, session, cfg, exit_frames=None, video_sec=0.0):
        """YOLO 박스를 읽고 ID를 붙인 뒤, 안전구역 안인지 위험인지를 매긴다."""
        import cv2
        import numpy as np
        cam = r['cam']
        kind = cam.get('kind') or 'coco'
        model = self.model_for(kind)
        prompt = cam.get('prompt')
        if prompt and self.yoloe_prompt != prompt:
            if not hasattr(self, 'prompt_cache'):
                self.prompt_cache = {}
            embedding = self.prompt_cache.get(prompt)
            if embedding is None:
                embedding = model.get_text_pe([prompt])
                self.prompt_cache[prompt] = embedding
            model.set_classes(list(cam['watch']), embedding)
            self.yoloe_prompt = prompt
        if r['tracker_session']!=session:
            r['box_tracker'].reset()
            r['tracker_session']=session
        ids=[i for i,name in model.names.items() if name in cam['watch']]
        if not ids:raise ValueError('No supported watch classes')
        result=model.predict(frame,classes=ids,conf=cam.get('conf', self.args.conf),
                             imgsz=cam.get('imgsz') or self.args.imgsz,
                             device=self.args.device,verbose=False)[0]
        raw=[]
        for box in result.boxes if result.boxes is not None else []:
            xyxy=[int(v) for v in box.xyxy[0].tolist()]
            if not accept_box(cam, xyxy):
                continue
            raw.append(dict(name=model.names[int(box.cls.item())],
                            confidence=float(box.conf.item()),
                            bbox=xyxy))
        # 매달린 판다는 전체 화면에서 점수가 낮다. 없을 때만 나무 쪽을 확대해 0.30 이상만 받는다.
        if 'panda' in cam.get('watch', ()) and not raw:
            height, width = frame.shape[:2]
            panda_conf = cam.get('conf', 0.3)
            for xa, xb, ya, yb in ((0.48, 0.68, 0.10, 0.62), (0.38, 0.78, 0.12, 0.90)):
                if raw:
                    break
                y0, y1 = int(height * ya), int(height * yb)
                x0, x1 = int(width * xa), int(width * xb)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                extra = model.predict(crop, classes=ids, conf=panda_conf,
                                      imgsz=cam.get('imgsz') or self.args.imgsz,
                                      device=self.args.device, verbose=False)[0]
                for box in extra.boxes if extra.boxes is not None else []:
                    xy = [int(v) for v in box.xyxy[0].tolist()]
                    xyxy = [xy[0] + x0, xy[1] + y0, xy[2] + x0, xy[3] + y0]
                    if not accept_box(cam, xyxy):
                        continue
                    raw.append(dict(name='panda', confidence=float(box.conf.item()), bbox=xyxy))
        remembered = []
        # 울타리 밖 기린은 전체 화면에서 빠지거나 사람으로만 잡힌다. 최근 아래쪽 화면을 다시 본다.
        if 'giraffe' in cam.get('watch', ()) and cam.get('alert_on_exit'):
            extra, remembered = self.fence_exit_marks(
                model, frame, video_sec, exit_frames or [], cfg)
            for det in extra:
                if accept_box(cam, det['bbox']):
                    raw.append(det)
        raw = suppress_overlaps(raw)
        h,w=frame.shape[:2];detections=[]
        poly=self.pixel_polygon(cfg,w,h)
        enabled=bool(cfg and cfg.get('enabled') and poly is not None)
        margin=int(cam.get('bottom_margin') or 0)
        air_top=int(cam.get('air_top') or 0)
        for det in r['box_tracker'].update(raw, air_top):
            name=det['name'];x1,y1,x2,y2=det['bbox']
            fx,fy=self.original.zs.anchor_xy(det['bbox'],h)
            point=(fx/max(w-1,1),fy/max(h-1,1))
            inside=bool(self.original.zs.in_poly(fx,fy,poly)) if enabled else None
            at_bottom=bool(margin and fy>=h-margin)
            zone,inside,danger=classify_subject(
                name,inside,at_bottom,enabled=enabled,
                alert_on_exit=bool(cam.get('alert_on_exit')),
                person_inside=bool(cfg and cfg.get('person_inside')))
            if air_top and y1 < air_top and name != 'person':
                zone, inside, danger = 'DANGER', False, True
            detections.append(dict(track_id=det['track_id'],name=name,confidence=det['confidence'],
                                   bbox=det['bbox'],
                                   norm=(x1/max(w-1,1), y1/max(h-1,1), x2/max(w-1,1), y2/max(h-1,1)),
                                   anchor=point,inside=inside,danger=danger,zone=zone,
                                   at_bottom=at_bottom))
        return detections, remembered

    def pixel_polygon(self, cfg, w, h):
        """저장된 0~1 좌표 ROI를 현재 프레임의 픽셀 다각형으로 되돌린다."""
        import numpy as np
        if not cfg or len(cfg.get('polygon') or [])<3:
            return None
        return (np.asarray(cfg['polygon'],np.float32)*[w-1,h-1]).astype(np.int32)

    def observe(self,cid,detections,frame,stamp,observed_at,video_sec,cfg):
        """락을 잡은 상태에서 호출된다. 경과 시간은 처리 시각이 아니라 프레임이 찍힌 시각으로 센다.
        기린 카메라에서 SAFE→DANGER로 바뀌면 1초를 기다리지 않고 바로 알림을 만든다.
        """
        r=self.runtime[cid];tracks=r['tracks']
        person_present=any(d['name']=='person' for d in detections)
        for d in detections:
            tid=d['track_id'];s=tracks.get(tid)
            # 다섯 영상을 돌아가며 보므로, 한 카메라의 다음 분석까지 수 초가 걸려도 같은 개체로 유지한다.
            if s and (stamp-s['last']>max(self.args.max_gap, 20) or s['name']!=d['name']):
                if s['event']:self.db.end(s['event'],'track_gap_or_class_change')
                del tracks[tid];s=None
            if s is None:
                s=dict(last=stamp,name=d['name'],event=None,revision=0,danger=False,was_safe=False,
                       pending=None,safe_since=None,history=deque(maxlen=64),
                       last_submit=-1e9,observed_signature=None,accepted_signature=None,
                       inflight=False,retries=0,zone='SAFE')
                tracks[tid]=s
            if r['cam'].get('latch_danger') and s.get('zone')=='DANGER' and not d['danger']:
                d=dict(d,danger=True,zone='DANGER',inside=False)
            before=s.get('zone','SAFE')
            after=d.get('zone') or ('DANGER' if d['danger'] else 'SAFE')
            # 안전에서 위험으로 바뀐 순간만 즉시 경고. 안전 복귀 뒤 alerted에서 빠지면 다시 즉시 경고한다.
            exited=bool(self.original.zs.is_zone_exit(before, after) and tid not in r['alerted'] and d['danger'])
            s['zone']=after
            s['last']=stamp
            if s['danger']!=d['danger']:
                s['revision']+=1  # 위험 여부가 바뀌면 이미 보낸 JEV 요청은 즉시 무효다.
                s['inflight']=False
                self.db.invalidate(s['event'])
            s['danger']=d['danger']
            if d['danger']:
                s['safe_since']=None
                if s['pending'] is None:s['pending']=stamp;s['history'].clear()
                s['history'].append((stamp,d['confidence']))
                if exited:
                    # 동물 이탈과 사람 진입은 문구가 다르다. 배너는 ALERT_HOLD_SEC 동안 유지한다.
                    r['alerted'].add(tid)
                    r['banner_text']=PERSON_ALERT_MSG if d['name']=='person' else self.original.zs.ALERT_MSG
                    r['banner_until']=time.monotonic()+self.original.zs.ALERT_HOLD_SEC
                confirmed=len(s['history'])>=2 and stamp-s['pending']>=self.args.confirm_sec
                if s['event'] is None and (exited or confirmed):
                    span=stamp-s['pending']
                    if d['name']=='person':
                        kind,message,zone='PERSON_INTRUSION',PERSON_ALERT_MSG,'안전구역_안'
                    elif exited:
                        kind,message,zone='ZONE_EXIT_DANGER',self.original.zs.ALERT_MSG,'철창밖_위험구역'
                    else:
                        kind='ZONE_EXIT' if s['was_safe'] else 'OUTSIDE_OBSERVED'
                        message,zone=None,'ROI'
                    s['event']=self.db.add_event(cid,tid,d['name'],d['confidence'],kind,frame,message,zone,
                                                 judge_severity(kind,d['confidence'],span))
                if s['event']:
                    self.maybe_submit(r,cid,tid,s,d,cfg,observed_at,video_sec,person_present,stamp)
            else:
                # 안전으로 돌아오면 위험 누적을 지운다. 0.8초 이상 안전하면 사건을 닫고 재경고를 허용한다.
                s['pending']=None;s['history'].clear()
                if s['safe_since'] is None:s['safe_since']=stamp
                if stamp-s['safe_since']>=.8:
                    s['was_safe']=True
                    r['alerted'].discard(tid)
                    if s['event']:self.db.end(s['event'],'safe_return');s['event']=None
                    s['accepted_signature']=None
                    s['inflight']=False
                    s['retries']=0
        # 이번 프레임에 안 보인 트랙이 너무 오래되면 놓친 것으로 보고 사건을 닫는다.
        for tid,s in list(tracks.items()):
            if stamp-s['last']>self.args.max_gap:
                if s['event']:self.db.end(s['event'],'track_lost_not_confirmed_safe')
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
        proxy=False
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

    def _shown_clip(self, runtime):
        """새로고침마다 같은 장수만큼만 앞으로 보낸다. 늦게 온 틱이 최신 장면으로 뛰지 않는다."""
        clips = runtime['clips']
        if not clips:
            return runtime.get('preview'), runtime['seq']
        fps = runtime.get('fps') or 24
        step = max(1, int(round(fps / 12)))
        newest = clips[-1][0]
        oldest = clips[0][0]
        cursor = runtime.get('play_cursor')
        if cursor is None:
            cursor = newest
        else:
            gap = newest - cursor
            if gap > int(fps * 1.5):
                cursor = newest
            elif gap > step:
                cursor += step + (1 if gap > step * 2 else 0)
            else:
                cursor += step
        if cursor > newest:
            cursor = newest
        if cursor < oldest:
            cursor = oldest
        runtime['play_cursor'] = cursor
        chosen = clips[0]
        for item in clips:
            if item[0] <= cursor:
                chosen = item
            else:
                break
        return chosen[1], chosen[0]

    def _live_boxes(self, runtime, seq, session):
        """마지막 두 탐지 위치로 지금 장면의 박스를 예상한다. 오래되면 그리지 않는다."""
        rendered = runtime.get('rendered')
        if not rendered or rendered[1] != session:
            return ()
        fps = runtime.get('fps') or 24
        max_age = int(fps * 1.2)
        boxes = []
        for hist in runtime['track_motion'].values():
            if not hist:
                continue
            seq_b, norm_b, tid, name, conf, danger, zone = hist[-1]
            age = seq - seq_b
            if age < -2 or age > max_age:
                continue
            norm = norm_b
            if len(hist) >= 2 and age > 0:
                seq_a, norm_a = hist[-2][0], hist[-2][1]
                dt = seq_b - seq_a
                if 0 < dt <= int(fps * 2):
                    k = min(age / dt, 1.25)
                    norm = tuple(max(0.0, min(1.0, norm_b[i] + (norm_b[i] - norm_a[i]) * k)) for i in range(4))
            elif age > int(fps * 0.35):
                continue
            boxes.append((norm, tid, name, conf, danger, zone))
        return boxes

    def step_all(self):
        """일정한 속도로 장면을 넘기고, 탐지 박스는 동물이 움직인 만큼 따라 그린다."""
        import cv2
        import numpy as np
        now = time.monotonic()
        with self.lock:
            shown = []
            signature = []
            for runtime in self.runtime.values():
                preview, seq = self._shown_clip(runtime)
                rendered = runtime['rendered']
                boxes = self._live_boxes(runtime, seq, runtime['session'])
                shown.append((preview, seq, deepcopy(runtime['cfg']), boxes,
                              runtime['banner_text'], runtime['banner_until']))
                det_seq = rendered[3] if rendered and len(rendered) > 3 else -1
                signature.append((seq, det_seq, runtime['session'], (runtime['cfg'] or {}).get('version'),
                                  runtime['banner_text'], bool(runtime['banner_text'] and now < runtime['banner_until']),
                                  len(boxes)))
            signature = tuple(signature)
            cached = getattr(self, '_view_cache', None)
            if signature == getattr(self, '_view_sig', None) and cached is not None:
                return cached
        out = []
        for preview, seq, cfg, boxes, banner_text, banner_until in shown:
            if preview is None:
                out.append(None)
                continue
            canvas = preview.copy()
            h, w = canvas.shape[:2]
            if cfg and cfg.get('polygon'):
                pts = (np.asarray(cfg['polygon']) * [w - 1, h - 1]).astype(np.int32)
                cv2.polylines(canvas, [pts], True, (0, 220, 230), 2)
            cv2.putText(canvas, roi_caption(cfg), (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 230), 2)
            for norm, tid, name, conf, danger, zone in boxes:
                nx1, ny1, nx2, ny2 = norm
                x1, y1 = int(nx1 * (w - 1)), int(ny1 * (h - 1))
                x2, y2 = int(nx2 * (w - 1)), int(ny2 * (h - 1))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                color = (0, 0, 255) if danger else (0, 210, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{name} {conf:.2f} #{tid} {zone}",
                            (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            if banner_text and now < banner_until:
                self.original.zs.draw_alert_banner(canvas, banner_text)
            out.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        self._view_sig = signature
        self._view_cache = out
        return out

    def note_timeline(self, cid, video_sec, detections):
        """파일 시각별로 박스를 남긴다. 영상이 반복돼도 그 시각의 박스는 그대로 쓴다."""
        try:
            stamp = round(float(video_sec or 0), 2)
        except (TypeError, ValueError):
            return
        fresh = [
            dict(tid=int(det['track_id']), name=det['name'], conf=round(float(det['confidence']), 3),
                 danger=bool(det['danger']), zone=det.get('zone') or '',
                 norm=[round(float(v), 4) for v in det['norm']])
            for det in detections if det.get('norm')]
        samples = self.timeline.setdefault(cid, [])
        for old in samples:
            if abs(old['t'] - stamp) > 0.1:
                continue
            merged = list(old['boxes'])
            for box in fresh:
                match = None
                for other in merged:
                    if other['name'] == box['name'] and norm_iou(other['norm'], box['norm']) >= 0.3:
                        match = other
                        break
                if match is None:
                    merged.append(box)
                    continue
                danger = bool(match['danger'] or box['danger'])
                if box['conf'] >= match['conf']:
                    match.update(box)
                match['danger'] = danger
                if danger:
                    match['zone'] = 'DANGER'
            old['boxes'] = merged
            old['t'] = stamp
            return
        samples.append(dict(t=stamp, boxes=fresh))
        if len(samples) > 1 and samples[-1]['t'] < samples[-2]['t']:
            samples.sort(key=lambda sample: sample['t'])
        if len(samples) > 800:
            del samples[:-800]

    def boxes_payload(self):
        """브라우저 오버레이가 읽는 탐지 기록."""
        out = []
        with self.lock:
            for index, cam in enumerate(self.original.CAMERAS):
                runtime = self.runtime[cam['cam_id']]
                polygon = ((runtime.get('cfg') or {}).get('polygon')) or []
                out.append(dict(i=index, polygon=polygon,
                                 samples=list(self.timeline.get(cam['cam_id'], []))))
        return out

    def open_stream(self):
        """브라우저가 원본 영상을 직접 재생하고, 박스는 JSON으로만 받는다."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        engine = self
        engine.media_files = [Path(cam['file']) for cam in self.original.CAMERAS]

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                    return

            def _cors(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_HEAD(self):
                self._route(write=False)

            def do_GET(self):
                self._route(write=True)

            def _route(self, write):
                path = self.path.split('?', 1)[0].rstrip('/') or '/'
                if path == '/boxes':
                    body = json.dumps(engine.boxes_payload(), separators=(',', ':')).encode('utf-8')
                    self.send_response(200)
                    self._cors()
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    if write:
                        self.wfile.write(body)
                    return
                if path.startswith('/media/'):
                    try:
                        index = int(path.rsplit('/', 1)[-1])
                    except ValueError:
                        self.send_error(404)
                        return
                    self._send_media(index, write)
                    return
                self.send_error(404)

            def _send_media(self, index, write):
                files = engine.media_files
                if index < 0 or index >= len(files) or not files[index].is_file():
                    self.send_error(404)
                    return
                path = files[index]
                size = path.stat().st_size
                start, end = 0, size - 1
                status = 200
                header = self.headers.get('Range')
                if header and header.startswith('bytes='):
                    spec = header.split('=', 1)[1].split(',', 1)[0].strip()
                    left, _, right = spec.partition('-')
                    try:
                        if left == '':
                            length = int(right or '0')
                            start = max(0, size - length)
                        else:
                            start = int(left)
                            end = int(right) if right else size - 1
                    except ValueError:
                        self.send_error(416)
                        return
                    if start >= size:
                        self.send_response(416)
                        self.send_header('Content-Range', f'bytes */{size}')
                        self.end_headers()
                        return
                    end = min(end, size - 1)
                    status = 206
                length = end - start + 1
                self.send_response(status)
                self._cors()
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(length))
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                if not write:
                    return
                try:
                    with path.open('rb') as handle:
                        handle.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = handle.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                    return

            def log_message(self, fmt, *args):
                return

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        httpd = None
        for offset in range(1, 8):
            port = self.args.port + offset
            try:
                httpd = Server(('127.0.0.1', port), Handler)
                self.stream_port = port
                break
            except OSError:
                httpd = None
        if httpd is None:
            self.stream_port = None
            return
        self.stream_httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True, name='cctv-http').start()

    def _clip_image(self, clips, seq):
        found = None
        for item_seq, image in clips:
            if item_seq == seq:
                return image
            if item_seq < seq:
                found = image
            else:
                break
        return found

    def _open_followers(self, frame, boxes):
        """탐지된 프레임에서 추적기를 켠다. 이후 장면마다 박스만 동물에 붙여 옮긴다."""
        import cv2
        height, width = frame.shape[:2]
        followers = []
        for norm, tid, name, conf in boxes:
            x1 = int(norm[0] * max(width - 1, 1))
            y1 = int(norm[1] * max(height - 1, 1))
            x2 = int(norm[2] * max(width - 1, 1))
            y2 = int(norm[3] * max(height - 1, 1))
            box_w = max(2, x2 - x1)
            box_h = max(2, y2 - y1)
            x1 = min(max(0, x1), max(0, width - box_w - 1))
            y1 = min(max(0, y1), max(0, height - box_h - 1))
            tracker = cv2.TrackerMIL_create()
            try:
                tracker.init(frame, (int(x1), int(y1), int(box_w), int(box_h)))
            except cv2.error:
                continue
            followers.append(dict(tracker=tracker, bbox=(x1, y1, box_w, box_h),
                                  tid=tid, name=name, conf=conf))
        return followers

    def _draw_followers(self, canvas, cfg, cam, native, followers):
        """추적 중인 박스를 지금 장면에 그린다. 발 위치로 안전/위험을 다시 매긴다."""
        import cv2
        import numpy as np
        height, width = canvas.shape[:2]
        native_h = native[1] if native else height
        scale = height / max(native_h, 1)
        poly = None
        if cfg and len(cfg.get('polygon') or []) >= 3:
            poly = (np.asarray(cfg['polygon'], np.float32) * [width - 1, height - 1]).astype(np.int32)
        enabled = bool(cfg and cfg.get('enabled') and poly is not None and len(poly) >= 3)
        margin = int(cam.get('bottom_margin') or 0) * scale
        air_top = int(cam.get('air_top') or 0) * scale
        for fol in followers:
            x, y, bw, bh = fol['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            fx, fy = (x1 + x2) / 2, y2
            inside = bool(self.original.zs.in_poly(fx, fy, poly)) if enabled else None
            at_bottom = bool(margin and fy >= height - margin)
            zone, _inside, danger = classify_subject(
                fol['name'], inside, at_bottom, enabled=enabled,
                alert_on_exit=bool(cam.get('alert_on_exit')),
                person_inside=bool(cfg and cfg.get('person_inside')))
            if air_top and y1 < air_top and fol['name'] != 'person':
                zone, danger = 'DANGER', True
            color = (0, 0, 255) if danger else (0, 210, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{fol['name']} {fol['conf']:.2f} #{fol['tid']} {zone}",
                        (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)

    def _render_camera(self, state, pending, clips, fps, cfg, cam, native, banner, banner_until, session, now):
        """파일 시계에 맞춰 한 장만 고르고, 탐지 박스는 그 사이의 동물을 따라가게 한다."""
        import cv2
        import numpy as np
        if not clips:
            return None, None
        fps = fps or 24.0
        consumed = None
        if pending and pending[0] == session:
            det_seq, boxes = pending[1], pending[2]
            det_frame = None
            if det_seq < clips[0][0]:
                consumed = det_seq
            else:
                for item_seq, image in clips:
                    if item_seq == det_seq:
                        det_frame = image
                        break
            if det_frame is not None:
                state['followers'] = self._open_followers(det_frame, boxes)
                state['synced'] = det_seq
                cursor = state.get('cursor')
                gap = 0 if cursor is None else cursor - det_seq
                if cursor is None or gap > int(fps * 1.5) or gap < -int(fps):
                    state['origin'] = (now, det_seq)
                    state['cursor'] = det_seq
                consumed = det_seq
        newest = clips[-1][0]
        oldest = clips[0][0]
        if state.get('origin') is None:
            state['origin'] = (now, oldest)
        start_t, start_seq = state['origin']
        target = start_seq + max(0, int((now - start_t) * fps))
        if target > newest:
            target = newest
        if target < oldest:
            target = oldest
        cursor = state.get('cursor')
        if cursor is None:
            cursor = target
        elif target > cursor + 2:
            target = cursor + 2
        elif target < cursor:
            target = cursor
        synced = state.get('synced')
        followers = state.get('followers') or []
        if followers and synced is not None and target > synced:
            limit = min(target, synced + 4)
            seq = synced
            while seq < limit:
                seq += 1
                nxt = None
                for item_seq, image in clips:
                    if item_seq == seq:
                        nxt = image
                        break
                if nxt is None:
                    state['synced'] = seq
                    continue
                for fol in followers:
                    try:
                        ok, box = fol['tracker'].update(nxt)
                    except cv2.error:
                        ok = False
                    if ok and box[2] > 1 and box[3] > 1:
                        fol['bbox'] = box
                state['synced'] = seq
        state['cursor'] = target
        shown = self._clip_image(clips, target)
        if shown is None:
            return None, consumed
        canvas = shown.copy()
        if cfg and cfg.get('polygon'):
            height, width = canvas.shape[:2]
            pts = (np.asarray(cfg['polygon']) * [width - 1, height - 1]).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (0, 220, 230), 2)
        cv2.putText(canvas, roi_caption(cfg), (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 230), 2)
        if state.get('synced') is not None and state['synced'] >= target - 2:
            self._draw_followers(canvas, cfg, cam, native, followers)
        if banner and now < banner_until:
            self.original.zs.draw_alert_banner(canvas, banner)
        ok, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return (buf.tobytes() if ok else None), consumed

    def stream_frames(self):
        """카메라 영상을 초당 20장, 매번 같은 간격으로 만들어 브라우저에 넘긴다."""
        states = {
            cid: {'session': None, 'origin': None, 'cursor': None, 'synced': None, 'followers': []}
            for cid in self.runtime
        }
        period = 1 / 20
        next_at = time.monotonic()
        while not self.stop.is_set():
            now = time.monotonic()
            jobs = []
            with self.lock:
                for cid, runtime in self.runtime.items():
                    state = states[cid]
                    if state['session'] != runtime['session']:
                        state.update(session=runtime['session'], origin=None, cursor=None,
                                     synced=None, followers=[])
                    jobs.append((cid, state, runtime.get('pending_follow'), list(runtime['clips']),
                                 runtime.get('fps') or 24.0, deepcopy(runtime['cfg']), runtime['cam'],
                                 runtime.get('native_size'), runtime['banner_text'], runtime['banner_until'],
                                 runtime['session']))
            jpegs = []
            consumed = []
            for cid, state, pending, clips, fps, cfg, cam, native, banner, banner_until, session in jobs:
                jpeg, seq = self._render_camera(state, pending, clips, fps, cfg, cam, native,
                                                 banner, banner_until, session, now)
                jpegs.append(jpeg)
                if seq is not None:
                    consumed.append((cid, seq))
            if consumed:
                with self.lock:
                    for cid, seq in consumed:
                        pending = self.runtime[cid].get('pending_follow')
                        if pending and pending[1] == seq and pending[0] == self.runtime[cid]['session']:
                            self.runtime[cid]['pending_follow'] = None
            with self.stream_lock:
                self.stream_jpegs = jpegs
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                if self.stop.wait(delay):
                    break
            else:
                next_at = time.monotonic()

    def close(self):
        """서버가 끝날 때 스레드를 멈추고 남은 사건을 종료한다."""
        self.stop.set()
        if self.stream_httpd is not None:
            self.stream_httpd.shutdown()
            self.stream_httpd.server_close()
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


def accept_box(cam, xyxy):
    """project_test와 같이 너무 작거나, 너무 넓거나, 통나무를 닮은 박스를 버린다."""
    width = xyxy[2] - xyxy[0]
    height = max(1, xyxy[3] - xyxy[1])
    if cam.get('min_height') and height < cam['min_height']:
        return False
    if cam.get('max_height') and height > cam['max_height']:
        return False
    if cam.get('max_aspect') and width / height > cam['max_aspect']:
        return False
    if cam.get('max_width') and width > cam['max_width']:
        return False
    return True


def box_area(xyxy):
    """박스 넓이. 두 마리를 한 박스로 덮은 경우를 가려낼 때 쓴다."""
    return max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])


def mostly_inside(outer, inner):
    """작은 박스 면적의 대부분이 큰 박스 안에 있으면 True."""
    x1, y1 = max(outer[0], inner[0]), max(outer[1], inner[1])
    x2, y2 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = max(1, box_area(inner))
    return inter / inner_area >= 0.75


def overlaps_same_body(a, b):
    """한 마리 위에 겹친 박스인지. 앞뒤로 선 다른 개체는 그대로 둔다."""
    if a['name'] != b['name']:
        return False
    if box_iou(a['bbox'], b['bbox']) >= 0.35:
        return True
    ax1, ay1, ax2, ay2 = a['bbox']
    bx1, by1, bx2, by2 = b['bbox']
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
    overlap_x = overlap_w / min(aw, bw)
    overlap_y = overlap_h / min(ah, bh)
    # 같은 x 위치에 위아래로 갈라진 박스는 한 마리다.
    if overlap_x >= 0.85 and overlap_y >= 0.2:
        return True
    center_gap = abs((ay1 + ay2) / 2 - (by1 + by2) / 2)
    if center_gap > 0.5 * min(ah, bh):
        return False
    return overlap_x >= 0.55 and overlap_y >= 0.25


def foot_outside(xyxy, poly, height):
    """발(박스 아래 가운데)이 안전구역 밖이거나, 구역이 없으면 화면 아래쪽이면 True."""
    import cv2
    fx = (xyxy[0] + xyxy[2]) / 2.0
    fy = float(min(xyxy[3], height - 1))
    if poly is None:
        return fy >= height * 0.82
    return cv2.pointPolygonTest(poly, (fx, fy), False) < 0


def fence_head(xyxy, width, height):
    """울타리 위로 나온 기린 머리 크기만 통과시킨다. 하단 글자나 바위는 버린다."""
    box_w = xyxy[2] - xyxy[0]
    box_h = xyxy[3] - xyxy[1]
    center_x = (xyxy[0] + xyxy[2]) / 2
    if not (14 <= box_w <= 55 and 26 <= box_h <= 75):
        return False
    if xyxy[1] < height * 0.74 or xyxy[3] > height - 28:
        return False
    return 0.55 * width <= center_x <= 0.78 * width


def exit_timeline_det(det, width, height):
    """과거 프레임에서 찾은 이탈 박스를 오버레이 기록 형식으로 만든다."""
    x1, y1, x2, y2 = det['bbox']
    scale_x = max(width - 1, 1)
    scale_y = max(height - 1, 1)
    return dict(track_id=8700, name='giraffe', confidence=det['confidence'],
                danger=True, zone='DANGER',
                norm=(x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y))


def norm_iou(a, b):
    """0~1 좌표 박스의 IoU."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = overlap_w * overlap_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def suppress_overlaps(detections, iou_min=0.45):
    """같은 개체에 겹친 박스는 신뢰도가 높은 것만 남긴다.
    한 마리를 덮은 작은 박스가 있으면, 그 위를 크게 덮어 사이 공간까지 포함한 박스는 버린다.
    """
    ranked = sorted(detections, key=lambda det: det['confidence'], reverse=True)
    kept = []
    for det in ranked:
        if any(overlaps_same_body(det, other) for other in kept):
            continue
        kept.append(det)
    pruned = []
    for det in kept:
        area = box_area(det['bbox'])
        covers_another = any(
            det is not other and det['name'] == other['name']
            and area > box_area(other['bbox']) * 1.6
            and mostly_inside(det['bbox'], other['bbox'])
            for other in kept)
        if not covers_another:
            pruned.append(det)
    return pruned


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

    def update(self, detections, air_top=0):
        """이전 박스와 가장 많이 겹치는 같은 클래스 박스에 기존 ID를 붙인다.
        사자가 울타리 위로 올라가면 겹침이 없어도 같은 기둥의 ID를 유지한다.
        """
        previous = set(self.tracks)
        pairs = []
        for tid, track in self.tracks.items():
            for index, det in enumerate(detections):
                if track['name'] != det['name']:
                    continue
                score = box_iou(track['bbox'], det['bbox'])
                if air_top:
                    old, new = track['bbox'], det['bbox']
                    old_cx, new_cx = (old[0] + old[2]) / 2, (new[0] + new[2]) / 2
                    old_cy, new_cy = (old[1] + old[3]) / 2, (new[1] + new[3]) / 2
                    half = max(old[2] - old[0], new[2] - new[0], 1) * 1.2
                    if abs(old_cx - new_cx) <= half and new_cy < old_cy:
                        score = max(score, self.iou_min)
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


def share_root():
    """이 스크립트가 포함된 공유 폴더. 그 상위 경로는 보지 않는다."""
    script_dir = Path(__file__).resolve().parent
    top = script_dir.parent
    for folder in (script_dir, top):
        if (folder / 'zoo_vision.py').is_file() and (folder / 'zoo_safety.py').is_file():
            return folder
    return top


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


def prepare_original(original):
    """project_test.py의 다섯 영상, ROI, 탐지 설정을 대시보드 카메라로 넣는다."""
    if not hasattr(original, 'zs'):
        import zoo_safety
        original.zs = zoo_safety
    video = Path(original.__file__).resolve().parent / 'video'
    specs = [
        ('CAMERA 01', '기린 방사장', 'ZOO_CAM_GIRAFFE_01', 'giraffe_cctv.mp4', ('giraffe', 'person'),
         [[40, 300], [180, 230], [420, 200], [700, 210], [830, 280], [820, 360], [650, 420], [430, 440], [180, 400], [50, 350]],
         dict(bottom_margin=8, conf=0.1, kind='coco', latch_danger=True, alert_on_exit=True, air_top=0, imgsz=960)),
        ('CAMERA 02', '사자 방사장', 'ZOO_CAM_LION_01', 'lion_cctv.mp4', ('lion',),
         [[0, 419], [174, 386], [599, 599], [553, 984], [0, 984]],
         dict(bottom_margin=0, conf=0.03, kind='yoloe', latch_danger=False, alert_on_exit=False, air_top=400,
              prompt='large cat lion', max_aspect=1.35, max_width=180)),
        ('CAMERA 03', '코끼리 방사장', 'ZOO_CAM_ELEPHANT_01', 'elephant_cctv.mp4', ('elephant',),
         [[0, 370], [420, 340], [780, 320], [1100, 340], [1279, 360], [1279, 640], [640, 650], [0, 660]],
         dict(bottom_margin=0, conf=0.25, kind='coco', latch_danger=True, alert_on_exit=False, air_top=0,
              min_height=60, max_aspect=1.6, imgsz=960)),
        ('CAMERA 04', '판다 방사장', 'ZOO_CAM_PANDA_01', 'panda_cctv.mp4', ('panda',),
         [[0, 640], [220, 540], [400, 440], [560, 340], [760, 250], [1000, 190], [1279, 160], [1279, 719], [0, 719]],
         dict(bottom_margin=0, conf=0.30, kind='yoloe', latch_danger=False, alert_on_exit=False, air_top=0,
              prompt='giant panda', max_width=340, max_height=420, imgsz=960)),
        ('CAMERA 05', '재규어 방사장', 'ZOO_CAM_JAGUAR_01', 'jaguar_cctv.mp4', ('jaguar',),
         [[0, 280], [480, 240], [800, 220], [1120, 200], [1279, 200], [1279, 719], [0, 719]],
         dict(bottom_margin=0, conf=0.03, kind='yoloe', latch_danger=False, alert_on_exit=False, air_top=0,
              prompt='ocelot', max_height=200, imgsz=960)),
    ]
    original.CAMERAS = [
        dict(code=code, title=title, cam_id=cam_id, file=video / filename, watch=watch,
             safe_zone=zone, live=True, **options)
        for code, title, cam_id, filename, watch, zone, options in specs
    ]


WALL_JS = r"""
const root = "http://127.0.0.1:__PORT__";
let pack = [];
function contentRect(video) {
  const ew = video.clientWidth, eh = video.clientHeight;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return {x:0, y:0, w:ew, h:eh};
  const scale = Math.min(ew / vw, eh / vh);
  const w = vw * scale, h = vh * scale;
  return {x:(ew - w) / 2, y:(eh - h) / 2, w:w, h:h};
}
function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}
function holdDanger(samples, t, boxes) {
  const extra = [];
  samples.forEach(function(s) {
    const age = t - s.t;
    if (age <= 0.04 || age > 2.8) return;
    (s.boxes || []).forEach(function(b) {
      if (!b.danger) return;
      const clash = boxes.concat(extra).some(function(o) { return boxIou(o.norm, b.norm) >= 0.2; });
      if (!clash) extra.push(b);
    });
  });
  return boxes.concat(extra);
}
function boxesAt(samples, t) {
  if (!samples || !samples.length) return [];
  let prev = null, next = null;
  for (let i = 0; i < samples.length; i++) {
    if (samples[i].t <= t) prev = samples[i];
    else { next = samples[i]; break; }
  }
  const maxGap = 3.0;
  let boxes = [];
  if (prev && next && (next.t - prev.t) <= 0.7 && (t - prev.t) <= maxGap && (next.t - t) <= maxGap) {
    const k = (t - prev.t) / Math.max(next.t - prev.t, 0.001);
    const used = {};
    (prev.boxes || []).forEach(function(b) {
      const n = (next.boxes || []).find(function(x) { return x.tid === b.tid && boxIou(b.norm, x.norm) >= 0.25; });
      if (n) {
        used[n.tid] = true;
        boxes.push({
          tid: b.tid, name: b.name, conf: b.conf,
          danger: k < 0.5 ? b.danger : n.danger,
          zone: k < 0.5 ? b.zone : n.zone,
          norm: b.norm.map(function(v, i) { return v + (n.norm[i] - v) * k; })
        });
      } else if (t - prev.t <= 0.3) boxes.push(b);
    });
    (next.boxes || []).forEach(function(n) {
      if (!used[n.tid] && next.t - t <= 0.3) boxes.push(n);
    });
  } else {
    const chosen = !prev ? next : (!next ? prev : ((t - prev.t) <= (next.t - t) ? prev : next));
    if (chosen && Math.abs(chosen.t - t) <= maxGap) boxes = (chosen.boxes || []).slice();
  }
  return holdDanger(samples, t, boxes);
}
function frame() {
  element.querySelectorAll("video").forEach(function(video) {
    const canvas = element.querySelector('canvas[data-i="' + video.dataset.i + '"]');
    if (!canvas) return;
    const w = video.clientWidth, h = video.clientHeight;
    if (w < 2 || h < 2) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const info = pack.find(function(item) { return String(item.i) === String(video.dataset.i); });
    const rect = contentRect(video);
    if (info && info.polygon && info.polygon.length >= 3) {
      ctx.beginPath();
      info.polygon.forEach(function(p, i) {
        const x = rect.x + p[0] * rect.w, y = rect.y + p[1] * rect.h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    const boxes = info ? boxesAt(info.samples, video.currentTime || 0) : [];
    boxes.forEach(function(b) {
      const x1 = rect.x + b.norm[0] * rect.w, y1 = rect.y + b.norm[1] * rect.h;
      const x2 = rect.x + b.norm[2] * rect.w, y2 = rect.y + b.norm[3] * rect.h;
      ctx.strokeStyle = b.danger ? "#ff2a2a" : "#22c55e";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "14px sans-serif";
      ctx.fillText(b.name + " " + Number(b.conf).toFixed(2) + " #" + b.tid + " " + (b.zone || ""), x1, Math.max(16, y1 - 6));
    });
  });
  requestAnimationFrame(frame);
}
function poll() {
  fetch(root + "/boxes").then(function(res) { return res.ok ? res.json() : []; }).then(function(data) {
    pack = data || [];
  }).catch(function() {}).finally(function() { setTimeout(poll, 300); });
}
element.querySelectorAll("video").forEach(function(video) { video.play().catch(function() {}); });
poll();
requestAnimationFrame(frame);
"""


def build_app(original,engine):
    """zoo_vision 대시보드의 데이터 함수를 이 엔진으로 갈아끼우고, JEV 탭과 ROI 편집을 붙인다."""
    import cv2
    import gradio as gr
    import numpy as np
    import pandas as pd
    # 원본 화면이 호출하는 함수들을, 파일 로그 대신 이 엔진의 DB/영상을 보게 바꾼다.
    original.ANALYZER=engine
    original.refresh_detect=engine.step_all
    original.load_events=lambda: pd.DataFrame(engine.db.events(),columns=original.EVENT_COLS)
    original.load_inbox=lambda: '\n'.join(
        f"[{r['detected_at']}] {r['camera_id']} #{r['track_id']} {r['message']}"
        for r in engine.db.events()[-30:]) or '오늘 알림 없음'
    original.latest_snapshots=lambda n=6: [r['snapshot_path'] for r in engine.db.events()[-n:][::-1]]

    def status(df):
        """오른쪽 카메라 상태 문장. 오류, 영상 끊김, ROI 없음, 감시 중을 구분한다."""
        alive=bool(engine.analyze_thread and engine.analyze_thread.is_alive())
        if engine.boot_error:
            thread='분석 스레드: 시작 실패 — '+engine.boot_error
        elif not alive:
            thread='분석 스레드: 중지됨'
        elif engine.last_ok:
            thread=f'분석 스레드: 동작 중 · 마지막 성공 {time.monotonic()-engine.last_ok:.1f}초 전'
        else:
            thread='분석 스레드: 동작 중 · 아직 성공한 분석 없음'
        lines=['### 카메라 상태',thread,f'진행 중 규칙 경고: **{engine.db.active_count()}건**']
        with engine.lock:
            for r in engine.runtime.values():
                cfg=r['cfg'];age=time.monotonic()-r['stamp']
                state=r['error'] or ('영상 대기/끊김' if age>engine.args.max_gap else
                    'ROI 미설정' if not cfg or not cfg['polygon'] else
                    '감시 중' if cfg['enabled'] else 'ROI 감시 꺼짐')
                lines.append(f"- {r['cam']['title']}: {state} · 분석 {r['analysis_fps']:.1f} FPS")
        return '\n'.join(lines)
    original.camera_status=status
    def alert_banner(df):
        """상단 경고. 지금 떠 있는 배너 문구를 우선하고, 없으면 오늘 마지막 사건을 보여 준다."""
        with engine.lock:
            live=next((r['banner_text'] for r in engine.runtime.values()
                       if r['banner_text'] and time.monotonic()<r['banner_until']), '')
        if live:
            return '<div class="alert-panel"><h3>'+live+'</h3></div>'
        rows=engine.db.events()
        if rows:
            last=rows[-1]
            return ('<div class="alert-panel"><h3>'
                    f"[{last.get('detected_at','')}] {last.get('severity','')} · {last.get('camera_id','')} · "
                    f"{last.get('object','')} {last.get('event_type','')}</h3><p>{last.get('message','')}</p></div>")
        return ('<div class="alert-panel"><h3>진행 중 경고 없음 — 영상·분석 연결 상태를 함께 확인하세요.</h3></div>')
    original.alert_banner=alert_banner

    def roi_frame(label):
        """편집에 쓸 현재 프레임. 없으면 None."""
        cam=next(c for c in original.CAMERAS if c['code']==label);r=engine.runtime[cam['cam_id']]
        with engine.lock:
            frame=None if r['frame'] is None else r['frame'].copy()
            cfg=deepcopy(r['cfg'])
        return frame,cfg

    def roi_preview(label,points):
        """찍은 점을 현재 화면 위에 다시 그린다."""
        import cv2
        frame,cfg=roi_frame(label)
        if frame is None:
            return None
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        return paint_clicks(rgb,points)

    def roi_open(label):
        """선택한 카메라 화면을 열고, 저장된 구역이 있으면 그 꼭짓점을 보여 준다."""
        frame,cfg=roi_frame(label)
        if frame is None:
            return None,[],False,False,False,'영상이 준비되지 않았습니다.'
        h,w=frame.shape[:2]
        points=[[float(x)*(w-1),float(y)*(h-1)] for x,y in cfg['polygon']] if cfg and cfg.get('polygon') else []
        has_zone=bool(points)
        note=('가장자리를 따라 클릭하세요. 점은 찍은 순서대로 이어집니다. '
              '3개 이상이면 저장할 수 있고, 잘못 찍으면 마지막 점만 취소하면 됩니다.')
        return (roi_preview(label,points),points,
                bool(cfg['enabled']) if has_zone else True,
                bool(cfg and cfg['person_inside']),bool(cfg and cfg['reviewed']),
                f'점 {len(points)}개. '+note)

    def roi_click(evt, points, label):
        """화면을 클릭한 곳에 다음 꼭짓점을 추가한다."""
        points=list(points or [])
        frame,_=roi_frame(label)
        if frame is None:
            return None,points,'영상이 준비되지 않았습니다. 현재 화면 불러오기를 다시 누르세요.'
        h,w=frame.shape[:2]
        idx=getattr(evt,'index',None)
        if not isinstance(idx,(list,tuple)) or len(idx)<2:
            return roi_preview(label,points),points,'클릭 위치를 읽지 못했습니다.'
        x,y=float(idx[0]),float(idx[1])
        if not (0<=x<w and 0<=y<h):
            return roi_preview(label,points),points,'화면 안쪽을 클릭하세요.'
        points.append([x,y])
        note='점 {}개.'.format(len(points))
        if len(points)<3:
            note+=' 가장자리를 따라 계속 클릭하세요.'
        elif polygon_crosses(points):
            note+=' 선이 교차합니다. 마지막 점을 취소하고 가장자리 순서로 다시 찍으세요.'
        else:
            note+=' 이 구역을 저장할 수 있습니다.'
        return roi_preview(label,points),points,note

    def roi_undo(points,label):
        """마지막으로 찍은 점만 뺀다."""
        points=list(points or [])[:-1]
        frame,_=roi_frame(label)
        if frame is None:
            return None,points,'영상이 준비되지 않았습니다.'
        return roi_preview(label,points),points,f'점 {len(points)}개. 마지막 점을 취소했습니다.'

    def roi_clear(points,label):
        """찍던 점을 모두 지운다. 저장하기 전까지 감시 중인 구역은 바뀌지 않는다."""
        frame,_=roi_frame(label)
        if frame is None:
            return None,[],'영상이 준비되지 않았습니다.'
        return roi_preview(label,[]),[],'점을 모두 지웠습니다. 가장자리부터 다시 클릭하세요.'

    def roi_save(label,points,enabled,person_inside,reviewed):
        """클릭한 꼭짓점을 안전구역으로 저장하고 그 카메라 추적을 리셋한다."""
        cam=next(c for c in original.CAMERAS if c['code']==label);r=engine.runtime[cam['cam_id']]
        try:
            points=list(points or [])
            with engine.lock:
                if r['frame'] is None:raise ValueError('영상이 준비되지 않았습니다.')
                if len(points)<3:raise ValueError('가장자리를 3곳 이상 클릭한 뒤 저장하세요.')
                if polygon_crosses(points):raise ValueError('선이 교차합니다. 마지막 점을 취소하고 가장자리 순서로 다시 찍으세요.')
                h,w=r['frame'].shape[:2]
                pts=np.array([[int(round(p[0])),int(round(p[1]))] for p in points],np.int32)
                if abs(cv2.contourArea(pts.astype(np.float32)))<10:raise ValueError('구역이 너무 좁습니다. 더 넓게 찍어 주세요.')
                version=r['roi_version']+1
                cfg=dict(polygon=(pts/[w-1,h-1]).tolist(),enabled=bool(enabled),
                         person_inside=bool(person_inside),reviewed=bool(reviewed),version=version)
                with engine.db.connect() as db:
                    db.execute('INSERT OR REPLACE INTO roi VALUES(?,?)',(cam['cam_id'],json.dumps(cfg)))
                engine.reset(r,'roi_changed');r['cfg']=cfg;r['roi_version']=version
                engine.timeline.pop(cam['cam_id'], None)
            return 'ROI 저장 완료. 이전 JEV 판단은 새 구역에 재사용하지 않습니다.'
        except (ValueError,TypeError) as exc:return str(exc)

    def jev_view():
        """JEV 탭: 최신 판단을 한글 문장으로 보여주고, 원본 JSON은 접어 둔다."""
        import html
        titles={c['cam_id']:c['title'] for c in original.CAMERAS}
        records=engine.db.latest();rows=[]
        cards=[]
        for item in records[:12]:
            state=json.loads(item['state_json']) if item.get('state_json') else {}
            response=json.loads(item['result_json']) if item.get('result_json') else {}
            answers=response.get('answers') or {}
            headline,lines,name,where,choice_ko=jev_brief(state,item.get('status'),answers)
            place=titles.get(state.get('source',{}).get('camera_id'), state.get('source',{}).get('camera_id','카메라'))
            ref=reference_label(item,engine.args.jev_ttl)
            when=str(item.get('requested_at') or '')[11:19]
            rows.append([when,place,name,where,STATUS_KO.get(item.get('status'), item.get('status')),
                         choice_ko,ref,'진행 중' if item.get('event_active') else '종료됨'])
            if len(cards)<1:
                bullets=''.join(f'<li>{html.escape(line)}</li>' for line in lines)
                badge='jev-now' if ref=='현재 참고' else 'jev-old'
                err=html.escape(item.get('error') or '')
                err_html=f'<p class="jev-error">{err}</p>' if err else ''
                cards.append(
                    '<div class="jev-card">'
                    f'<p class="jev-kicker">{html.escape(place)} · {html.escape(when)} · '
                    f'<span class="{badge}">{html.escape(ref)}</span></p>'
                    f'<h2>{html.escape(headline)}</h2>'
                    f'<ul>{bullets}</ul>{err_html}'
                    '<p class="jev-note">이 판단은 참고입니다. 규칙 경고를 끄거나 설비를 움직이지 않습니다.</p>'
                    '</div>')
        if engine.args.jev and getattr(engine.args,'provider','')=='ollama':
            mode=f'로컬 모델 {engine.args.jev_model}에 판단을 요청합니다.'
        elif engine.args.jev:
            mode='외부 API에 판단을 요청합니다.'
        else:
            mode='지금은 JSON만 만들고, 밖으로 보내지 않습니다.'
        story=(cards[0] if cards else
               '<div class="jev-card"><h2>아직 판단할 사건이 없습니다</h2>'
               '<p>동물이 구역을 벗어나거나 사람이 구역 안에 있으면 여기에 한글로 요약됩니다.</p></div>')
        story=f'<p class="jev-mode">{html.escape(mode)}</p>'+story
        newest=records[0] if records else None
        request={'model':engine.args.jev_model,'state':json.loads(newest['state_json']),
                 'questions':questions(engine.args.score)} if newest else {}
        response=json.loads(newest['result_json']) if newest and newest['result_json'] else {}
        return story,rows,request,response

    # 화면: project_test의 다섯 영상. 통계·사건 표는 원본 레이아웃. JEV 판단 탭만 여기 추가한다.
    with gr.Blocks(title='SafeZoo · YOLO + JEV') as demo:
        gr.HTML(original.header_html());gr.HTML(original.hero_html())
        gr.Markdown('영상 재생·YOLO 분석은 백그라운드에서 동작합니다. JEV에는 사건 요약 JSON만 전송합니다.')
        banner=gr.HTML()
        with gr.Tabs():
            with gr.Tab('실시간 감시'):
                import html as html_lib
                if engine.stream_port:
                    cards=[]
                    for index, cam in enumerate(original.CAMERAS):
                        title=html_lib.escape(cam['title'])
                        src=f'http://127.0.0.1:{engine.stream_port}/media/{index}'
                        cards.append(
                            '<figure style="margin:0;background:#111;border-radius:12px;overflow:hidden;">'
                            '<div style="position:relative;height:280px;background:#000;">'
                            f'<video data-i="{index}" src="{src}" muted autoplay loop playsinline '
                            'style="width:100%;height:280px;object-fit:contain;background:#000;display:block;"></video>'
                            f'<canvas data-i="{index}" style="position:absolute;left:0;top:0;width:100%;height:280px;pointer-events:none;"></canvas>'
                            '</div>'
                            f'<figcaption style="color:#e4e4e7;padding:6px 10px 8px;font-size:14px;">{title}</figcaption></figure>')
                    wall_js = WALL_JS.replace('__PORT__', str(engine.stream_port))
                    gr.HTML('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 12px;">'
                            +''.join(cards)+'</div>',
                            apply_default_css=False, js_on_load=wall_js)
                else:
                    gr.Markdown('영상 스트림 포트를 열지 못했습니다. 7864번 근처 포트를 비운 뒤 다시 실행하세요.')
                points=gr.State([])
                with gr.Accordion('ROI 찍기',open=True):
                    camera=gr.Dropdown([c['code'] for c in original.CAMERAS],value='CAMERA 01',label='카메라')
                    get=gr.Button('현재 화면 불러오기')
                    board=gr.Image(label='가장자리를 클릭하세요',type='numpy',interactive=False,
                                   height=420,buttons=['fullscreen'])
                    with gr.Row():
                        undo=gr.Button('마지막 점 취소')
                        clear=gr.Button('점 모두 지우기')
                        save=gr.Button('이 구역 저장')
                    enabled=gr.Checkbox(value=True,label='구역 감시 활성화')
                    person=gr.Checkbox(label='사람이 ROI 안에 있으면 경고')
                    reviewed=gr.Checkbox(label='운영자가 ROI 경계를 검토함')
                    message=gr.Markdown('카메라를 고른 뒤 화면을 불러오고, 안전구역 가장자리를 순서대로 클릭하세요. 안쪽을 칠할 필요는 없습니다.')
                    get.click(roi_open,camera,[board,points,enabled,person,reviewed,message])
                    camera.change(roi_open,camera,[board,points,enabled,person,reviewed,message])
                    # gr.SelectData 힌트는 이 함수가 함수 안에 있어 자동으로 인식되지 않는다.
                    roi_click.__annotations__={'evt':gr.SelectData}
                    board.select(roi_click,[points,camera],[board,points,message])
                    undo.click(roi_undo,[points,camera],[board,points,message])
                    clear.click(roi_clear,[points,camera],[board,points,message])
                    save.click(roi_save,[camera,points,enabled,person,reviewed],message)
                cam_md=gr.Markdown();live_kpi=gr.Markdown();inbox=gr.Textbox(label='오늘 기본 알림',lines=5)
                gallery=gr.Gallery(label='오늘 스냅샷',columns=3,height=230)
            kpi,cam_stat,chart_type,chart_sev=original.build_stats_tab()
            kw,button,table,inbox_full=original.build_events_tab()
            with gr.Tab('JEV 판단'):
                with gr.Column(elem_id='jev-panel'):
                    story=gr.HTML()
                    decision_table=gr.Dataframe(headers=['시각','장소','본 것','어디','전송','판단','참고','사건'],
                                                interactive=False,elem_classes=['jev-table'])
                    gr.Markdown('맨 위 문장이 가장 최근 판단입니다. **현재 참고**만 지금 진행 중인 사건에 해당하고, '
                                '**당시 기록**은 이미 끝난 상황입니다.',
                                elem_classes=['panel-box'])
                    with gr.Accordion('원본 JSON',open=False):
                        request_json=gr.JSON(label='보낸 내용')
                        response_json=gr.JSON(label='받은 답 (없으면 비어 있음)')
        outputs=[banner,live_kpi,kpi,cam_md,cam_stat,chart_type,chart_sev,table,inbox,inbox_full,gallery]
        demo.load(original.refresh_all,kw,outputs)
        button.click(original.refresh_all,kw,outputs);kw.change(original.refresh_all,kw,outputs)
        gr.Timer(2).tick(original.refresh_all,kw,outputs)
        demo.load(jev_view,None,[story,decision_table,request_json,response_json])
        gr.Timer(2).tick(jev_view,None,[story,decision_table,request_json,response_json])
    return demo


def parse_args():
    """실행 옵션. 신뢰도 기본값은 0.1이라 점수가 낮은 기린·판다 박스도 남긴다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=None)  # 비우면 공유 폴더(zoo_vision.py가 있는 곳)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'zoo_jev_data_new')  # 이 파일 전용 SQLite와 스냅샷
    parser.add_argument('--weights',type=Path)  # 비우면 yolo11n.pt를 찾는다
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
    root = share_root()
    if args.source_dir is None:
        args.source_dir = root
    if args.weights is None:
        choices = [
            root / 'yolo11n.pt',
            Path(__file__).resolve().parent / 'yolo11n.pt',
            root / 'unit5' / 'yolo11n.pt',
        ]
        args.weights = next((item for item in choices if item.is_file()), None)
        if args.weights is None:
            parser.error('yolo11n.pt를 공유 폴더 안에 두세요: ' + str(root))
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
.cctv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 0 0 12px; }
.cctv-card { margin: 0; background: #111; border-radius: 12px; overflow: hidden; }
.cctv-card img { width: 100%; height: 280px; object-fit: contain; background: #000; display: block; }
.cctv-card figcaption { color: #e4e4e7; padding: 6px 10px 8px; font-size: 14px; }
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
    prepare_original(original)
    engine=LinkedAnalyzer(original,args)
    engine.open_stream()
    demo=build_app(original,engine)
    engine.start()
    stream=f' | video: http://127.0.0.1:{engine.stream_port}' if engine.stream_port else ' | video stream failed'
    print(f'SafeZoo: http://127.0.0.1:{args.port}{stream} | decision: {args.provider if args.jev else "JSON only"} / {args.jev_model}')
    try:
        demo.launch(server_name='127.0.0.1',server_port=args.port,share=args.share,
                    auth=('zoo',os.environ['SAFEZOO_PASSWORD']) if args.share else None,
                    theme=original.build_theme(),css=original.load_css()+JEV_CSS,
                    allowed_paths=[str(engine.db.root/'snapshots')],
                    blocked_paths=[str(engine.db.path)])
    finally:
        engine.close()


if __name__=='__main__':
    main()  # python zoo_vision_jev.py 로 직접 실행할 때만 서버를 연다.
