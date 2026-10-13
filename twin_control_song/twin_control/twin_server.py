"""ZOO-GIS Twin Control 연동 서버.

zoo_safe/zoo_vision_jev_new.py의 분석 엔진(LinkedAnalyzer)을 그대로 돌리고,
그 결과를 twin_control.html 관제 화면에 넘긴다. Gradio 화면은 띄우지 않는다.

    /                              관제 화면
    /stream/{camera_id}.mjpg       분석 박스가 그려진 영상 (MJPEG)
    /ws                            카메라 상태와 오늘 사건 목록 (0.5초 주기)
    /api/state                     같은 내용의 1회 조회
    /api/events?day=YYYY-MM-DD     하루치 사건 목록 (비우면 오늘)
    /api/events.csv?day=...        같은 내용의 CSV 파일
    /api/ai/judge                  하루치 미확인 사건을 이 PC의 Ollama TEV에 일괄 판단 요청 (POST)
    /api/ai/job                    일괄 판단 진행 상황
    /api/events/{id}/ack           사건 확인
    /api/events/{id}/review        검토 구분 기록 (실제 위험 / 오탐 / 판단 보류)
    /api/events/{id}/snapshot      사건 순간의 증거 사진
    /api/cameras/{id}/frame.jpg    박스 없는 현재 화면 (구역 편집 배경)
    /api/cameras/{id}/roi          안전구역 조회(GET)와 저장(POST)

기본은 이 PC(127.0.0.1)에서만 열린다. 사건 DB는 기존 대시보드와 섞이지 않게 twin_data/에 따로 둔다.

다른 PC에서 접속하게 하려면 계정을 먼저 만들고 --host 0.0.0.0 으로 실행한다.
    python twin_server.py --add-user 이름 --role admin     (비밀번호는 입력창에서 받는다)
    python twin_server.py --list-users / --remove-user 이름
    python twin_server.py --host 0.0.0.0
계정이 하나라도 있으면 모든 화면과 영상이 로그인을 요구한다. 계정 없이 --host를 바꾸면 실행을 거부한다.
admin은 구역 편집까지, operator는 사건 확인·검토와 AI 판단 요청까지 할 수 있다.
암호화(HTTPS)는 하지 않는다. 같은 건물의 내부망에서만 쓰고, 인터넷에 직접 열지 않는다.
여기에 없는 옵션(--weights, --conf, --device 등)은 분석 엔진으로 그대로 넘어간다.
--ollama를 붙이면 진행 중인 사건마다 TEV 판단을 실시간으로 받는다. 붙이지 않아도 일괄 판단은 된다.
AI 판단은 참고 정보다. 사건의 확인 상태와 규칙 경고를 바꾸지 않는다.

실행:
    & c:\\code\\code_test\\zoo\\.venv\\Scripts\\python.exe twin_server.py
"""
import argparse
import asyncio
from contextlib import asynccontextmanager, contextmanager
import csv
import getpass
import hashlib
import hmac
import io
import json
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import threading
import time
import types
import urllib.error
import uuid
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

import twin_vision
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response,
                               StreamingResponse)

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent / 'zoo_safe'
PAGE = HERE / 'twin_control.html'
PAGE_HEAD = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>')
FRAME_SEC = 0.2    # 영상 속도를 아직 모를 때 쓰는 화면 갱신 간격
STATE_SEC = 0.5    # 상태 전송 주기
EVENT_LIMIT = 60   # 화면에 넘기는 오늘 사건 수 (최근 것부터)
EVENT_ID = re.compile(r'^[0-9a-f]{32}$')
REVIEW_LABELS = ('실제 위험', '오탐', '판단 보류')
DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')
USER_NAME = re.compile(r'^[A-Za-z0-9가-힣_.-]{2,32}$')
ROLES = ('admin', 'operator')
PASSWORD_MIN = 10
COOKIE = 'twin_session'
SESSION_SEC = 12 * 3600   # 한 근무 교대 길이
LOGIN_TRIES = 5           # 이만큼 틀리면 그 주소는 잠시 막는다
LOGIN_LOCK_SEC = 60
LOOPBACK = ('127.0.0.1', 'localhost', '::1')
LOGIN_PAGE = """<title>ZOO-GIS Twin Control 로그인</title>
<style>
html,body{height:100%;margin:0;background:#0a111e;color:#e7edf7;font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;font-size:14px}
body{display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box}
form{width:100%;max-width:340px;background:#101a2d;border:1px solid #24334f;border-radius:10px;padding:24px;display:flex;flex-direction:column;gap:12px}
h1{margin:0 0 4px;font-size:18px;letter-spacing:.02em}
label{display:flex;flex-direction:column;gap:5px;color:#8d9ab4;font-size:12.5px}
input{font:inherit;color:#e7edf7;background:#16223a;border:1px solid #31436a;border-radius:6px;padding:9px 10px}
button{font:inherit;font-weight:600;color:#06101f;background:#4c9dff;border:0;border-radius:6px;padding:10px;cursor:pointer}
:focus-visible{outline:2px solid #4c9dff;outline-offset:2px}
#msg{min-height:20px;color:#f2b33d;font-size:12.5px}
</style>
<form id="f">
  <h1>ZOO-GIS TWIN CONTROL</h1>
  <label for="name">이름<input id="name" autocomplete="username" required autofocus></label>
  <label for="password">비밀번호<input id="password" type="password" autocomplete="current-password" required></label>
  <button>로그인</button>
  <div id="msg" role="alert"></div>
</form>
<script>
document.getElementById('f').addEventListener('submit', function(ev){
  ev.preventDefault();
  var msg=document.getElementById('msg'); msg.textContent='';
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:document.getElementById('name').value,password:document.getElementById('password').value})})
    .then(function(r){ return r.json().then(function(d){ return {ok:r.ok,d:d}; }); })
    .then(function(res){ if(res.ok) location.href='/'; else msg.textContent=res.d.detail||'로그인하지 못했습니다.'; })
    .catch(function(){ msg.textContent='서버에 연결하지 못했습니다.'; });
});
</script>"""
BATCH_LIMIT = 300        # 일괄 판단 한 번에 보내는 최대 사건 수
BATCH_TIMEOUT = 90       # 첫 호출은 모델을 올리느라 오래 걸린다
AI_ANSWERED = ('completed', 'stale', 'batch_completed')
AI_WAITING = ('pending', 'running', 'batch_running')
AI_FAILED = ('error', 'unavailable', 'batch_error')
SEVERITY_EN = {'경고': 'warning', '주의': 'caution', '참고': 'info'}
CSV_FIELDS = ('detected_at', 'camera_id', 'track_id', 'object', 'zone', 'event_type', 'severity',
              'message', 'confidence', 'active', 'ended_at', 'end_reason', 'acknowledged_at',
              'ack_name', 'review_label', 'review_note', 'snapshot_path')


class Accounts:
    """로그인 계정. 사건 DB와 섞이지 않게 accounts.sqlite3에 따로 둔다. 비밀번호는 scrypt 해시만 저장한다."""

    def __init__(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(root) / 'accounts.sqlite3'
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS users('
                       'name TEXT PRIMARY KEY, role TEXT, salt BLOB, hash BLOB, created_at TEXT)')

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _hash(password, salt):
        return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)

    def users(self):
        with self._db() as db:
            return db.execute('SELECT name, role, created_at FROM users ORDER BY name').fetchall()

    def add(self, name, role, password):
        if not USER_NAME.match(name):
            raise ValueError('이름은 2~32자의 한글, 영문, 숫자, 밑줄, 점, 하이픈만 쓸 수 있습니다.')
        if role not in ROLES:
            raise ValueError('역할은 ' + ', '.join(ROLES) + ' 중 하나입니다.')
        if len(password) < PASSWORD_MIN:
            raise ValueError(f'비밀번호는 {PASSWORD_MIN}자 이상이어야 합니다.')
        salt = secrets.token_bytes(16)
        with self._db() as db:
            db.execute('INSERT INTO users(name, role, salt, hash, created_at) VALUES(?,?,?,?,?) '
                       'ON CONFLICT(name) DO UPDATE SET role=excluded.role, salt=excluded.salt, hash=excluded.hash',
                       (name, role, salt, self._hash(password, salt), time.strftime('%Y-%m-%d %H:%M:%S')))

    def remove(self, name):
        with self._db() as db:
            return db.execute('DELETE FROM users WHERE name=?', (name,)).rowcount

    def check(self, name, password):
        """맞으면 역할, 틀리면 None. 없는 이름도 같은 시간만큼 해시를 계산한다."""
        with self._db() as db:
            row = db.execute('SELECT role, salt, hash FROM users WHERE name=?', (name,)).fetchone()
        salt, stored = (row[1], row[2]) if row else (b'\0' * 16, b'')
        ok = hmac.compare_digest(self._hash(password, salt), stored)
        return row[0] if row and ok else None


def own_options(argv):
    """이 서버의 옵션만 먼저 읽는다. 나머지는 분석 엔진의 parse_args에 맡긴다."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'twin_data')
    parser.add_argument('--add-user')
    parser.add_argument('--role', choices=ROLES, default='operator')
    parser.add_argument('--remove-user')
    parser.add_argument('--list-users', action='store_true')
    return parser.parse_known_args(argv)


def account_command(own):
    """계정 명령이면 처리하고 True. 비밀번호는 명령줄이 아니라 입력창에서 받는다."""
    if not (own.add_user or own.remove_user or own.list_users):
        return False
    accounts = Accounts(own.data_dir)
    if own.add_user:
        first = getpass.getpass(f'{own.add_user} 비밀번호 ({PASSWORD_MIN}자 이상): ')
        if first != getpass.getpass('한 번 더 입력: '):
            raise SystemExit('두 입력이 다릅니다. 계정을 만들지 않았습니다.')
        try:
            accounts.add(own.add_user, own.role, first)
        except ValueError as exc:
            raise SystemExit(str(exc))
        print(f'계정 저장: {own.add_user} ({own.role}). 서버를 다시 시작하면 적용됩니다.')
    if own.remove_user:
        print('계정 삭제:' if accounts.remove(own.remove_user) else '없는 계정:', own.remove_user)
    if own.list_users:
        rows = accounts.users()
        print('등록된 계정이 없습니다. 이 상태에서는 로그인 없이 이 PC에서만 열립니다.' if not rows
              else '\n'.join(f'{name}\t{role}\t{created}' for name, role, created in rows))
    return True


class Guard:
    """로그인하지 않은 요청을 앱에 닿기 전에 돌려보낸다. 영상 스트림을 감싸지 않도록 ASGI 층에서 본다."""

    def __init__(self, app, check):
        self.app, self.check = app, check

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            response = self.check(Request(scope))
            if response is not None:
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_engine(own, rest):
    sys.path.insert(0, str(ENGINE_DIR))
    import zoo_vision_jev_new as zj
    saved = sys.argv
    sys.argv = [saved[0], '--data-dir', str(own.data_dir), *rest]
    try:
        args = zj.parse_args()
    finally:
        sys.argv = saved
    original = zj.load_original(args.source_dir)
    # 카메라 목록과 탐지 후처리는 twin_vision에 있다. 원본 zoo_vision.py는 실행 중에만 바꿔 끼운다.
    original.CAMERAS[:] = twin_vision.engine_cameras(original, args.source_dir)
    return zj, twin_vision.make_analyzer(zj)(original, args), args


def create_app(zj, engine, args, accounts):
    from dashboard_policy import END_REASON_KO

    stop = threading.Event()
    frames = {}
    snapshot_root = (engine.db.root / 'snapshots').resolve()
    live_ai = bool(args.jev and args.provider == 'ollama')
    ai_args = types.SimpleNamespace(
        provider='ollama', jev_model=args.jev_model if args.provider == 'ollama' else 'tev1:0.8b',
        ollama_url=args.ollama_url, jev_timeout=BATCH_TIMEOUT, score=False)
    job = dict(running=False, day='', total=0, done=0, errors=0, last_error='', finished_at='')
    job_lock = threading.Lock()

    def ai_view(row):
        """decisions 한 줄을 화면용으로 줄인다. 답도 대기도 실패도 아닌 기록(전송 안 함 등)은 None."""
        status = row['status']
        base = dict(status=status, model=row['model_requested'], at=row['requested_at'],
                    mode='batch' if status.startswith('batch') else 'live')
        if row['result_json'] and status in AI_ANSWERED:
            result = json.loads(row['result_json'])
            situation = result['answers']['situation']
            return dict(base, state='answered', choice=situation['choice'],
                        probability=round(situation['probabilities'][situation['choice']], 3),
                        needs_review=round(result['answers']['needs_immediate_review']['noul'], 3),
                        latency_ms=result.get('latency_ms'))
        if status in AI_WAITING:
            return dict(base, state='waiting')
        if status in AI_FAILED:
            return dict(base, state='failed', error=row['error'] or '')
        return None

    def ai_for(event_ids):
        """사건별 최신 AI 판단. 답이 있는 요청은 나중의 답 없는 요청에 밀리지 않는다."""
        found, ids = {}, list(event_ids)
        with engine.db.connect() as db:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                marks = ','.join('?' * len(chunk))
                for row in db.execute(
                        'SELECT event_id, status, result_json, error, model_requested, requested_at '
                        f'FROM decisions WHERE event_id IN ({marks}) ORDER BY rowid', chunk):
                    view = ai_view(row)
                    if view is None:
                        continue
                    old = found.get(row['event_id'])
                    if view['state'] == 'answered' or not (old and old['state'] == 'answered'):
                        found[row['event_id']] = view
        return found

    def with_ai(rows):
        answers = ai_for(row['id'] for row in rows)
        return [event_view(row, answers.get(row['id'])) for row in rows]

    def review_state(row, day_rows_):
        """끝난 사건을 나중에 판단시킬 때 보내는 요약. 사건 당시 저장해 둔 요약이 있으면 그것을 쓴다."""
        with engine.db.connect() as db:
            stored = db.execute('SELECT state_json FROM decisions WHERE event_id=? AND state_json IS NOT NULL '
                                'ORDER BY rowid DESC LIMIT 1', (row['id'],)).fetchone()
        cid, name = row.get('camera_id'), row.get('object')
        if stored:
            state = json.loads(stored[0])
        else:
            person = row.get('event_type') == 'PERSON_INTRUSION'
            state = dict(
                schema_version='1.0',
                source=dict(camera_id=cid, mode='recorded_video_demo', observed_at=row.get('detected_at')),
                subject=dict(detected_class=name,
                             class_is_proxy=twin_vision.is_proxy(cid, name),
                             detector_confidence_min=row.get('confidence')),
                evidence=dict(current_roi_relation='inside' if person else 'outside',
                              previously_observed_safe=row.get('event_type') == 'ZONE_EXIT_DANGER'),
                context=dict(authorized_work='unknown', physical_barrier_status='unknown',
                             person_identity='unknown'),
                policy=dict(version=zj.POLICY_VERSION, roi_meaning='permitted animal area',
                            animal_outside='alert when an animal foot is outside the ROI; not proof of escape',
                            person_outside='not an alert',
                            safety_rule='AI must not clear the rule alarm or control equipment. '
                                        'Missing information is not evidence of safety.'))
        at = row.get('detected_at') or ''
        window = [other for other in day_rows_
                  if other['id'] != row['id'] and other.get('camera_id') == cid
                  and (other.get('detected_at') or '') <= at
                  and _seconds(at) - _seconds(other.get('detected_at') or '') <= 600]
        state['review'] = dict(
            mode='post_hoc_review', event_still_active=bool(row['active']), end_reason=row['end_reason'],
            rule_event_type=row.get('event_type'), rule_severity=SEVERITY_EN.get(row.get('severity')),
            same_camera_events_previous_10min=len(window))
        return state

    def run_batch(day, targets, day_rows_):
        streak = 0
        for row in targets:
            if stop.is_set():
                break
            jid = uuid.uuid4().hex
            try:
                state = review_state(row, day_rows_)
                engine.db.record_request(dict(id=jid, event=row['id'], state=state), ai_args, 'batch_running')
                started = time.monotonic()
                response = zj.call_jev(state, ai_args)
                response['latency_ms'] = round((time.monotonic() - started) * 1000)
                engine.db.decision(jid, 'batch_completed', response)
                streak = 0
            except Exception as exc:
                message = ('HTTP ' + str(exc.code) if isinstance(exc, urllib.error.HTTPError)
                           else type(exc).__name__)
                try:
                    engine.db.decision(jid, 'batch_error', error=message)
                except Exception:
                    pass
                streak += 1
                with job_lock:
                    job['errors'] += 1
                    job['last_error'] = message
            with job_lock:
                job['done'] += 1
            if streak >= 3:
                with job_lock:
                    job['last_error'] = 'Ollama 응답 실패가 3번 이어져 중단했습니다 (' + job['last_error'] + ')'
                break
        with job_lock:
            job.update(running=False, finished_at=zj.now().isoformat())

    def play_interval():
        """카메라 영상 중 가장 짧은 프레임 간격. 화면은 이 속도로 보낸다."""
        with engine.lock:
            rates = [r.get('play_fps') or 0 for r in engine.runtime.values()]
        fps = max(rates) if rates else 0
        return 1 / fps if fps >= 1 else FRAME_SEC

    def pump():
        """방금 읽은 프레임을 JPEG로 바꿔 둔다. 새 프레임이 있을 때만 다시 그린다."""
        import cv2
        ids = list(engine.runtime)
        seen = {}
        while not stop.is_set():
            began = time.monotonic()
            try:
                with engine.lock:
                    current = {cid: engine.runtime[cid]['seq'] for cid in ids}
                if current != seen:
                    for cid, image in zip(ids, engine.step_all()):
                        if image is None:
                            continue
                        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            frames[cid] = buf.tobytes()
                    seen = current
            except Exception as exc:
                print('[frame]', type(exc).__name__, exc)
            remain = play_interval() - (time.monotonic() - began)
            stop.wait(remain if remain > 0 else 0.005)

    def event_view(row, ai=None):
        path = row.get('snapshot_path')
        return dict(
            id=row['id'], detected_at=row.get('detected_at'), camera_id=row.get('camera_id'),
            track_id=row.get('track_id'), object=row.get('object'),
            object_ko=twin_vision.label_of(zj.CLASS_KO.get(row.get('object'), row.get('object'))),
            event_type=row.get('event_type'), severity=row.get('severity'),
            message=row.get('message'), confidence=row.get('confidence'),
            active=row['active'], ended_at=row['ended_at'],
            end_reason_ko=END_REASON_KO.get(row['end_reason'], row['end_reason']),
            acknowledged_at=row['acknowledged_at'], ack_name=row['ack_name'],
            review_label=row['review_label'], review_note=row['review_note'], zone=row.get('zone'),
            has_snapshot=bool(path and Path(path).is_file()), ai=ai)

    def state():
        current = zj.now()
        cameras = []
        with engine.lock:
            for cid, r in engine.runtime.items():
                rendered, cfg = r['rendered'], r['cfg']
                fresh = bool(rendered and rendered[1] == r['session']
                             and time.monotonic() - rendered[0] <= args.max_gap)
                cameras.append(dict(
                    id=cid, title=r['cam']['title'], connected=r['frame'] is not None,
                    error=r['error'], fps=round(r['analysis_fps'], 1),
                    armed=bool(cfg and cfg.get('enabled')),
                    ignored=len(getattr(engine.model, 'ignored', {}).get(cid, [])),
                    detections=[dict(name=d['name'], label=twin_vision.label_of(d['name']),
                                     confidence=round(d['confidence'], 2),
                                     danger=bool(d['danger']), zone=d.get('zone') or '',
                                     track_id=d['track_id']) for d in rendered[2]] if fresh else []))
        rows = engine.db.list_records(day_from=current.date().isoformat(), limit=5000)
        return dict(now=current.isoformat(),
                    now_sec=current.hour * 3600 + current.minute * 60 + current.second,
                    boot_error=engine.boot_error, cameras=cameras,
                    events=with_ai(rows[-EVENT_LIMIT:]),
                    unacked=engine.db.unacked_count(),
                    ai=dict(live=live_ai, model=ai_args.jev_model, job=dict(job)))

    auth_on = bool(accounts.users())
    sessions, session_lock, failures = {}, threading.Lock(), {}

    def local_origin(headers):
        """다른 사이트의 스크립트가 이 서버를 건드리지 못하게, 요청 출처가 접속한 주소와 같은지 본다."""
        origin = headers.get('origin')
        return not origin or urlparse(origin).netloc == headers.get('host')

    def session_of(cookies):
        token = cookies.get(COOKIE)
        with session_lock:
            found = sessions.get(token) if token else None
            if found and found['until'] < time.time():
                sessions.pop(token, None)
                found = None
            return found

    def who(cookies):
        """요청한 사람. 계정이 없는 실행(이 PC 전용)에서는 모두 admin으로 본다."""
        if not auth_on:
            return dict(name='', role='admin')
        found = session_of(cookies)
        return dict(name=found['name'], role=found['role']) if found else None

    def need(request, *roles):
        me = who(request.cookies)
        if me is None:
            raise HTTPException(401, '로그인이 필요합니다.')
        if me['role'] not in roles:
            raise HTTPException(403, '이 작업은 ' + ' 또는 '.join(roles) + ' 계정만 할 수 있습니다.')
        return me

    def gate(request):
        path = request.url.path
        if not auth_on or path in ('/login', '/api/login') or session_of(request.cookies):
            return None
        if path.startswith(('/api/', '/stream/')):
            return JSONResponse({'detail': '로그인이 필요합니다.'}, status_code=401)
        return RedirectResponse('/login', status_code=303)

    def checked_id(event_id, headers):
        if not local_origin(headers):
            raise HTTPException(403, '이 PC의 관제 화면에서만 요청할 수 있습니다.')
        if not EVENT_ID.match(event_id):
            raise HTTPException(404, '사건을 찾을 수 없습니다.')
        return event_id

    @asynccontextmanager
    async def lifespan(_app):
        engine.start()
        thread = threading.Thread(target=pump, daemon=True, name='frame-pump')
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            engine.close()

    app = FastAPI(title='ZOO-GIS Twin Control', lifespan=lifespan)
    app.add_middleware(Guard, check=gate)

    @app.get('/login', response_class=HTMLResponse)
    def login_page():
        return PAGE_HEAD + LOGIN_PAGE

    @app.post('/api/login')
    async def login(request: Request):
        if not local_origin(request.headers):
            raise HTTPException(403, '이 관제 화면에서만 요청할 수 있습니다.')
        address, current = request.client.host if request.client else '', time.time()
        with session_lock:
            count, until = failures.get(address, (0, 0))
        if until > current:
            raise HTTPException(429, f'여러 번 틀렸습니다. {int(until - current) + 1}초 뒤에 다시 시도하세요.')
        body = await request.json()
        name = str(body.get('name') or '')[:64]
        role = await asyncio.to_thread(accounts.check, name, str(body.get('password') or '')[:256])
        with session_lock:
            if role is None:
                count += 1
                failures[address] = (0, current + LOGIN_LOCK_SEC) if count >= LOGIN_TRIES else (count, 0)
            else:
                failures.pop(address, None)
                token = secrets.token_urlsafe(32)
                sessions[token] = dict(name=name, role=role, until=current + SESSION_SEC)
        if role is None:
            raise HTTPException(401, '이름 또는 비밀번호가 맞지 않습니다.')
        response = JSONResponse({'name': name, 'role': role})
        response.set_cookie(COOKIE, token, max_age=SESSION_SEC, httponly=True, samesite='strict', path='/')
        return response

    @app.post('/api/logout')
    def logout(request: Request):
        with session_lock:
            sessions.pop(request.cookies.get(COOKIE), None)
        response = JSONResponse({'ok': True})
        response.delete_cookie(COOKIE, path='/')
        return response

    @app.get('/api/me')
    def me(request: Request):
        return dict(need(request, *ROLES), login=auth_on)

    @app.get('/', response_class=HTMLResponse)
    def page():
        return PAGE_HEAD + PAGE.read_text(encoding='utf-8')

    @app.get('/api/state')
    def api_state():
        return state()

    @app.websocket('/ws')
    async def ws(socket: WebSocket):
        if not local_origin(socket.headers):
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while not stop.is_set():
                me = who(socket.cookies)
                if me is None:
                    # 로그인이 없거나 끝났다. 화면이 로그인 창으로 가도록 따로 정한 코드로 닫는다.
                    await socket.close(code=4401)
                    return
                payload = await asyncio.to_thread(state)
                payload['me'] = dict(me, login=auth_on)
                await socket.send_json(payload)
                await asyncio.sleep(STATE_SEC)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.get('/stream/{camera_id}.mjpg')
    async def stream(camera_id: str):
        if camera_id not in engine.runtime:
            raise HTTPException(404, '카메라를 찾을 수 없습니다.')

        async def parts():
            sent = None
            while not stop.is_set():
                jpeg = frames.get(camera_id)
                if jpeg is not None and jpeg is not sent:
                    sent = jpeg
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                await asyncio.sleep(play_interval() / 2)

        return StreamingResponse(parts(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    @app.post('/api/events/{event_id}/ack')
    async def ack(event_id: str, request: Request):
        eid = checked_id(event_id, request.headers)
        me = need(request, *ROLES)
        body = await request.json()
        result = engine.db.acknowledge([eid], me['name'] or str(body.get('name') or '')[:40])
        if result['missing']:
            raise HTTPException(404, '사건을 찾을 수 없습니다.')
        if me['name'] and result['changed']:
            # 엔진은 확인자를 '미인증:이름'으로 적는다. 로그인한 사람은 그렇게 남기지 않는다.
            with engine.db.connect() as db:
                db.execute('UPDATE events SET ack_name=? WHERE id=?', ('로그인:' + me['name'], eid))
        return result

    @app.post('/api/events/{event_id}/review')
    async def review(event_id: str, request: Request):
        eid = checked_id(event_id, request.headers)
        me = need(request, *ROLES)
        body = await request.json()
        if body.get('label') not in REVIEW_LABELS:
            raise HTTPException(400, '검토 구분은 ' + ', '.join(REVIEW_LABELS) + ' 중 하나입니다.')
        note = str(body.get('note') or '')[:480] + (' (' + me['name'] + ')' if me['name'] else '')
        if not engine.db.review(eid, body['label'], note):
            raise HTTPException(404, '사건을 찾을 수 없습니다.')
        return {'reviewed': eid, 'label': body['label']}

    @app.get('/api/events/{event_id}/snapshot')
    def snapshot(event_id: str):
        if not EVENT_ID.match(event_id):
            raise HTTPException(404, '사건을 찾을 수 없습니다.')
        row = engine.db.record(event_id)
        path = Path(row['snapshot_path']).resolve() if row and row.get('snapshot_path') else None
        if path is None or not path.is_file() or not path.is_relative_to(snapshot_root):
            raise HTTPException(404, '증거 이미지 없음')
        return FileResponse(path, media_type='image/jpeg')

    def day_rows(day):
        day = day or zj.now().date().isoformat()
        if not DAY.match(day):
            raise HTTPException(400, '날짜는 YYYY-MM-DD 형식입니다.')
        return day, engine.db.list_records(day_from=day, day_to=day, archived='all', limit=100000)

    @app.get('/api/events')
    def events_of_day(day: str = ''):
        day, rows = day_rows(day)
        return dict(day=day, events=with_ai(rows))

    @app.get('/api/events.csv')
    def events_csv(day: str = ''):
        day, rows = day_rows(day)

        def cell(value):
            # 표 계산 프로그램이 수식으로 읽지 않게 한다.
            text = '' if value is None else str(value)
            return "'" + text if text[:1] in ('=', '+', '-', '@') else text

        out = io.StringIO()
        writer = csv.writer(out, lineterminator='\r\n')
        writer.writerow(CSV_FIELDS + ('ai_situation', 'ai_probability', 'ai_needs_review', 'ai_model', 'ai_mode'))
        answers = ai_for(row['id'] for row in rows)
        for row in rows:
            ai = answers.get(row['id'])
            ai = ai if ai and ai['state'] == 'answered' else {}
            writer.writerow([cell(int(row['active']) if name == 'active' else row.get(name)) for name in CSV_FIELDS]
                            + [cell(ai.get(key)) for key in ('choice', 'probability', 'needs_review', 'model', 'mode')])
        return Response(('\ufeff' + out.getvalue()).encode('utf-8'), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': f'attachment; filename="zoo_events_{day}.csv"',
                                 'Cache-Control': 'no-store'})

    @app.get('/api/ai/job')
    def ai_job():
        with job_lock:
            return dict(job)

    @app.post('/api/ai/judge')
    async def ai_judge(request: Request):
        """그날의 미확인 사건 가운데 아직 AI 답이 없는 것만 판단을 요청한다. 확인 상태는 바꾸지 않는다."""
        if not local_origin(request.headers):
            raise HTTPException(403, '이 관제 화면에서만 요청할 수 있습니다.')
        need(request, *ROLES)
        body = await request.json()
        day, rows = day_rows(str(body.get('day') or ''))
        answers = ai_for(row['id'] for row in rows)
        targets = [row for row in rows if not row['acknowledged_at']
                   and (answers.get(row['id']) or {}).get('state') != 'answered'][:BATCH_LIMIT]
        with job_lock:
            if job['running']:
                raise HTTPException(409, '이미 판단 요청이 진행 중입니다.')
            job.update(running=bool(targets), day=day, total=len(targets), done=0, errors=0,
                       last_error='', finished_at='' if targets else zj.now().isoformat())
            snapshot = dict(job)
        if targets:
            threading.Thread(target=run_batch, args=(day, targets, rows), daemon=True, name='ai-batch').start()
        return snapshot

    def camera(camera_id):
        if camera_id not in engine.runtime:
            raise HTTPException(404, '카메라를 찾을 수 없습니다.')
        return engine.runtime[camera_id]

    def roi_view(camera_id):
        with engine.lock:
            r = camera(camera_id)
            cfg = r['cfg'] or {}
            return dict(camera=camera_id, polygon=cfg.get('polygon') or [], enabled=bool(cfg.get('enabled')),
                        person_inside=bool(cfg.get('person_inside')), version=r['roi_version'],
                        ready=r['frame'] is not None)

    @app.get('/api/cameras/{camera_id}/roi')
    def roi_get(camera_id: str):
        return roi_view(camera_id)

    @app.get('/api/cameras/{camera_id}/frame.jpg')
    def frame_jpg(camera_id: str):
        import cv2
        with engine.lock:
            frame = camera(camera_id)['frame']
            frame = None if frame is None else frame.copy()
        if frame is None:
            raise HTTPException(503, '영상이 준비되지 않았습니다.')
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, '화면을 만들지 못했습니다.')
        return Response(buf.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.post('/api/cameras/{camera_id}/roi')
    async def roi_save(camera_id: str, request: Request):
        """zoo_safe 대시보드의 ROI 저장과 같은 절차. 좌표는 0~1 비율로 받는다."""
        import cv2
        import numpy as np
        if not local_origin(request.headers):
            raise HTTPException(403, '이 관제 화면에서만 요청할 수 있습니다.')
        need(request, 'admin')
        body = await request.json()
        try:
            points = [[float(p[0]), float(p[1])] for p in body.get('polygon') or []]
        except (TypeError, ValueError, IndexError):
            raise HTTPException(400, '구역 좌표를 확인하세요.')
        if len(points) < 3:
            raise HTTPException(400, '안전구역은 가장자리를 3곳 이상 찍은 뒤 저장하세요.')
        if len(points) > 64 or any(not 0 <= v <= 1 for p in points for v in p):
            raise HTTPException(400, '구역 좌표가 화면 범위를 벗어났습니다.')
        if zj.polygon_crosses(points):
            raise HTTPException(400, '선이 교차합니다. 가장자리 순서대로 다시 찍으세요.')
        with engine.lock:
            r = camera(camera_id)
            if r['frame'] is None:
                raise HTTPException(503, '영상이 준비되지 않았습니다.')
            height, width = r['frame'].shape[:2]
            pixels = (np.asarray(points, np.float32) * [width - 1, height - 1]).astype(np.float32)
            if abs(cv2.contourArea(pixels)) < 10:
                raise HTTPException(400, '구역이 너무 좁습니다.')
            before = r['cfg'] or {}
            version = r['roi_version'] + 1
            cfg = dict(polygon=points, enabled=bool(body.get('enabled')),
                       person_inside=bool(body.get('person_inside')), reviewed=True, version=version,
                       exclude_polygon=before.get('exclude_polygon') or [],
                       exclude_reviewed=bool(before.get('exclude_reviewed')))
            with engine.db.connect() as db:
                db.execute('INSERT INTO roi(camera, json) VALUES(?,?) '
                           'ON CONFLICT(camera) DO UPDATE SET json=excluded.json',
                           (camera_id, json.dumps(cfg)))
            engine.reset(r, 'roi_changed')
            r['cfg'] = cfg
            r['roi_version'] = version
        return roi_view(camera_id)

    return app


def _seconds(stamp):
    """'YYYY-MM-DD HH:MM:SS'의 하루 중 초. 형식이 다르면 0."""
    match = re.search(r'(\d\d):(\d\d):(\d\d)', stamp or '')
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) if match else 0


_CTRL_HANDLER = None


def restore_ctrl_c():
    """Windows에서 OpenCV·PyTorch가 Ctrl+C를 삼킨 뒤에도 종료 신호가 파이썬에 닿게 한다.

    두 라이브러리는 서버가 뜬 뒤에도 콘솔 종료 신호를 무시하도록 다시 건다.
    그래서 무시 설정을 주기적으로 지우고, 우리 처리기를 맨 앞에 다시 둔다.
    한 번 누르면 정상 종료, 한 번 더 누르면 바로 프로세스를 끝낸다.
    """
    global _CTRL_HANDLER
    if sys.platform != 'win32':
        return
    import _thread
    import ctypes
    from ctypes import wintypes
    import os
    import warnings
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    hits = {'n': 0}

    @routine
    def handler(ctrl_type):
        if ctrl_type not in (0, 1):
            return False
        hits['n'] += 1
        _thread.interrupt_main()
        if hits['n'] >= 2:
            os._exit(130)
        return True

    _CTRL_HANDLER = handler
    kernel32.SetConsoleCtrlHandler(None, False)
    kernel32.SetConsoleCtrlHandler(handler, True)

    def guard():
        while True:
            kernel32.SetConsoleCtrlHandler(None, False)
            kernel32.SetConsoleCtrlHandler(handler, False)
            kernel32.SetConsoleCtrlHandler(handler, True)
            time.sleep(0.5)

    threading.Thread(target=guard, daemon=True, name='ctrl-c').start()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main():
    import uvicorn
    own, rest = own_options(sys.argv[1:])
    if account_command(own):
        return
    accounts = Accounts(own.data_dir)
    users = accounts.users()
    if own.host not in LOOPBACK and not users:
        raise SystemExit('다른 PC에서 접속하게 하려면 계정이 필요합니다. 먼저 실행하세요:\n'
                         '  python twin_server.py --add-user 이름 --role admin')
    zj, engine, args = build_engine(own, rest)
    print(f'ZOO-GIS Twin Control: http://{"127.0.0.1" if own.host in LOOPBACK else "이 PC의 IP 주소"}:{own.port}')
    print(f'  접속 범위     : {"이 PC만" if own.host in LOOPBACK else "같은 네트워크의 다른 PC 포함 (" + own.host + ")"}')
    print(f'  로그인        : {"계정 " + str(len(users)) + "개, 로그인 필요" if users else "계정 없음, 로그인 없이 열림"}')
    print(f'  영상·원본 코드: {args.source_dir}')
    print(f'  사건 DB       : {engine.db.path}')
    print(f'  YOLO 가중치   : {args.weights}')
    restore_ctrl_c()
    uvicorn.run(create_app(zj, engine, args, accounts), host=own.host, port=own.port, log_level='warning')


if __name__ == '__main__':
    main()
