"""SafeZoo 대시보드: 기존 zoo_vision.py 화면에 YOLO 탐지와 JEV 판단을 붙인다.

한 줄 흐름:
  영상 읽기 → YOLO로 동물/사람 박스 → 같은 물체끼리 ID 유지
  → 기린 카메라는 zoo_safety.py 규칙으로 안전구역 이탈 즉시 알림
  → 위험 상태가 이어지면 JEV에 보낼 JSON을 만들고, --jev 또는 --ollama일 때만 전송

실행 (Anaconda 기본 환경):
  python zoo_vision_jev.py
실제 TypeSafe API: TYPESAFE_API_KEY 설정 후 python zoo_vision_jev.py --jev
로컬 Ollama: python zoo_vision_jev.py --ollama --score
외부 공개 링크: SAFEZOO_PASSWORD 설정 후 --share (아이디는 zoo)

원본 zoo_vision.py / zoo_safety.py 와 그 로그 파일은 수정하지 않는다.
영상·모델의 기본 위치는 C:\\code\\code_test\\zoo (--source-dir 로 변경 가능).
이 파일이 새로 쓰는 데이터는 같은 폴더의 zoo_jev_data/ (SQLite + 날짜별 스냅샷).
--jev 가 없어도 JSON은 만들고 기록만 한다. API 요청은 나가지 않는다.
아래 질문·정책 문장은 데모용이며, 실제 동물원 절차로 검증된 값이 아니다.
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
SCORE_LEVELS = [
    'The action conflicts with policy or could worsen the situation.',
    'The action offers little direct help with this event.',
    'The action could help but needs a missing prerequisite or another action first.',
    'The action directly addresses the event and is supported by policy and context.',
]


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

    def add_event(self, cid, tid, name, confidence, kind, frame, message=None, zone='ROI'):
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
                   track_id=tid,object=name,zone=zone,event_type=kind,severity='HIGH',
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
                continue
            if time.monotonic() < self.error_until:
                self.engine.db.decision(job['id'],'unavailable',error='API cooldown; base alarm retained')
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
            except Exception as exc:
                # API 키나 응답 본문 전체는 로그에 남기지 않고, 오류 종류만 기록한다.
                message = 'HTTP '+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__
                self.engine.db.decision(job['id'],'error',error=message+'; base alarm retained')
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
                alerted=set(),banner_text='',banner_until=0.)
        self.model = None
        self.worker = JevWorker(self)
        self.threads = []

    def start(self):
        """JEV 스레드, 카메라별 재생 스레드, YOLO 분석 스레드를 켠다."""
        self.worker.thread.start()
        for cid in self.runtime:
            t=threading.Thread(target=self.capture,args=(cid,),daemon=True)
            t.start(); self.threads.append(t)
        t=threading.Thread(target=self.analyze,daemon=True)
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
        import torch
        if self.args.device=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 1))
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
                        # 추론이 느려도 박스는 남긴다. 경고 판단만 늦은 프레임에서 건너뛴다.
                        finished=time.monotonic()
                        r['rendered']=(finished,session,detections)
                        r['analysis_fps']=1/max(finished-r['last_analysis'],.001) if r['last_analysis'] else 0
                        r['last_analysis']=finished
                        if finished-stamp>self.args.max_gap:
                            r['error']='분석 지연: 화면은 갱신하고 경고 판단은 보류'
                            continue
                        self.observe(cid,detections,frame,stamp,observed_at,video_sec,cfg)
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
        # 기린 카메라(alert_on_exit)는 zoo_safety.py와 같은 발 위치 규칙을 쓴다.
        use_safety=bool(r['cam'].get('alert_on_exit') and cfg and cfg.get('enabled') and poly is not None)
        for det in r['box_tracker'].update(raw):
            name=det['name'];x1,y1,x2,y2=det['bbox']
            point=((x1+x2)/2/max(w-1,1),min(y2,h-1)/max(h-1,1))
            if use_safety:
                zone,inside=self.safety_zone(det['bbox'],h,poly)
                danger=zone=='DANGER'
            else:
                # 다른 카메라는 사람 침입/동물 이탈을 옵션에 따라 따로 본다. 발 하단 규칙은 쓰지 않는다.
                inside=(cv2.pointPolygonTest(np.asarray(cfg['polygon'],np.float32),point,False)>=0
                        if cfg and cfg.get('polygon') else None)
                armed=bool(cfg and cfg.get('enabled') and inside is not None)
                danger=bool(armed and (inside if name=='person' and cfg['person_inside'] else
                                      False if name=='person' else not inside))
                zone='DANGER' if danger else 'SAFE' if inside is not None else ''
            detections.append(dict(track_id=det['track_id'],name=name,confidence=det['confidence'],
                                   bbox=det['bbox'],anchor=point,inside=inside,danger=danger,zone=zone))
        return detections

    def pixel_polygon(self, cfg, w, h):
        """저장된 0~1 좌표 ROI를 현재 프레임의 픽셀 다각형으로 되돌린다."""
        import numpy as np
        if not cfg or len(cfg.get('polygon') or [])<3:
            return None
        return (np.asarray(cfg['polygon'],np.float32)*[w-1,h-1]).astype(np.int32)

    def safety_zone(self, bbox, frame_h, polygon):
        """zoo_safety.py와 같은 판정: 발 위치, 하단 여백, 안전구역 다각형."""
        zs=self.original.zs
        fx,fy=zs.anchor_xy(bbox,frame_h)
        if fy>=frame_h-zs.BOTTOM_MARGIN:
            return 'DANGER', False
        inside=bool(zs.in_poly(fx,fy,polygon))
        return ('SAFE' if inside else 'DANGER'), inside

    def observe(self,cid,detections,frame,stamp,observed_at,video_sec,cfg):
        """락을 잡은 상태에서 호출된다. 경과 시간은 처리 시각이 아니라 프레임이 찍힌 시각으로 센다.
        기린 카메라에서 SAFE→DANGER로 바뀌면 1초를 기다리지 않고 바로 알림을 만든다.
        """
        r=self.runtime[cid];tracks=r['tracks']
        person_present=any(d['name']=='person' for d in detections)
        for d in detections:
            tid=d['track_id'];s=tracks.get(tid)
            # 같은 ID라도 오래 비었거나 클래스(기린↔사람)가 바뀌면 다른 물체로 본다.
            if s and (stamp-s['last']>self.args.max_gap or s['name']!=d['name']):
                if s['event']:self.db.end(s['event'],'track_gap_or_class_change')
                del tracks[tid];s=None
            if s is None:
                s=dict(last=stamp,name=d['name'],event=None,revision=0,danger=False,was_safe=False,
                       pending=None,safe_since=None,history=deque(maxlen=64),signature=None,
                       last_submit=-1e9,observed_signature=None,zone='SAFE')
                tracks[tid]=s
            before=s.get('zone','SAFE')
            after=d.get('zone') or ('DANGER' if d['danger'] else 'SAFE')
            # zoo_safety.is_zone_exit: 안전(또는 처음 봄)에서 위험으로 바뀐 순간. 같은 트랙은 한 번만.
            exited=bool(r['cam'].get('alert_on_exit') and self.original.zs.is_zone_exit(before, after)
                        and tid not in r['alerted'])
            s['zone']=after
            s['last']=stamp
            if s['danger']!=d['danger']:
                s['revision']+=1  # 위험 여부가 바뀌면 이미 보낸 JEV 요청은 즉시 무효다.
                s['signature']=None
                self.db.invalidate(s['event'])
            s['danger']=d['danger']
            if d['danger']:
                s['safe_since']=None
                if s['pending'] is None:s['pending']=stamp;s['history'].clear()
                s['history'].append((stamp,d['confidence']))
                if exited:
                    # 화면 배너 문구는 zoo_safety.ALERT_MSG 와 같다. ALERT_HOLD_SEC 동안 유지한다.
                    r['alerted'].add(tid)
                    r['banner_text']=self.original.zs.ALERT_MSG
                    r['banner_until']=time.monotonic()+self.original.zs.ALERT_HOLD_SEC
                confirmed=len(s['history'])>=2 and stamp-s['pending']>=self.args.confirm_sec
                if s['event'] is None and (exited or confirmed):
                    if exited:
                        kind,message,zone='ZONE_EXIT_DANGER',self.original.zs.ALERT_MSG,'철창밖_위험구역'
                    else:
                        kind=('PERSON_INTRUSION' if d['name']=='person' else 'ZONE_EXIT' if s['was_safe'] else 'OUTSIDE_OBSERVED')
                        message,zone=None,'ROI'
                    s['event']=self.db.add_event(cid,tid,d['name'],d['confidence'],kind,frame,message,zone)
                    s['signature']=None
                if s['event']:
                    # 매 프레임마다 JEV를 부르지 않는다. 사람 동석 여부나 낮은 신뢰도로 상태가 바뀔 때만.
                    signature=(person_present,d['confidence']<.5)
                    if signature!=s['observed_signature']:
                        s['revision']+=1
                        s['observed_signature']=signature
                        self.db.invalidate(s['event'])
                    if signature!=s['signature'] and stamp-s['last_submit']>=self.args.cooldown:
                        s['signature']=signature;s['last_submit']=stamp;s['revision']+=1
                        state=self.build_state(cid,d,s,cfg,observed_at,video_sec,person_present,stamp)
                        job=dict(id=uuid.uuid4().hex,event=s['event'],camera=cid,track=tid,
                                 session=r['session'],revision=s['revision'],created=time.monotonic(),state=state)
                        self.worker.submit(job)
            else:
                # 안전 구역으로 돌아오면 위험 누적을 지우고, 0.8초 이상 안전하면 사건을 닫는다.
                s['pending']=None;s['history'].clear()
                if s['safe_since'] is None:s['safe_since']=stamp
                if stamp-s['safe_since']>=.8:
                    s['was_safe']=True
                    if s['event']:self.db.end(s['event'],'safe_return');s['event']=None;s['signature']=None
        # 이번 프레임에 안 보인 트랙이 너무 오래되면 놓친 것으로 보고 사건을 닫는다.
        for tid,s in list(tracks.items()):
            if stamp-s['last']>self.args.max_gap:
                if s['event']:self.db.end(s['event'],'track_lost_not_confirmed_safe')
                del tracks[tid]

    def build_state(self,cid,d,s,cfg,observed_at,video_sec,person_present,stamp):
        """JEV에 보낼 사건 요약. 영상 전체가 아니라 탐지 결과와 구역 관계만 담는다."""
        samples=list(s['history']);gaps=[b[0]-a[0] for a,b in zip(samples,samples[1:])]
        # 표범은 cat/dog/bear, 판다는 bear로 대신 탐지한다. 그 경우 실제 종명이 아니다.
        proxy=('LEOPARD' in cid or 'PANDA' in cid) and d['name']!='person'
        state=dict(schema_version='1.0',source=dict(camera_id=cid,mode='recorded_video_demo',
            observed_at=observed_at,video_position_sec=round(video_sec,2),roi_version=cfg['version'],
            observation_age_ms=round((time.monotonic()-stamp)*1000)),
            subject=dict(detected_class=d['name'],class_is_proxy=proxy,
                         detector_confidence_min=round(min(c for _,c in samples),3)),
            evidence=dict(current_roi_relation='inside' if d['inside'] else 'outside',
                previously_observed_safe=s['was_safe'],violation_observed_span_sec=round(stamp-s['pending'],2),
                recent_violation_samples=len(samples),max_recent_gap_sec=round(max(gaps,default=0),2),
                person_detected_in_same_camera=person_present),
            context=dict(roi_reviewed_by_operator=cfg['reviewed'],authorized_work='unknown',
                         physical_barrier_status='unknown',person_identity='unknown'),
            policy=dict(version=POLICY_VERSION,roi_meaning='permitted animal area',
                animal_outside='request operator verification; not proof of escape',
                person_inside='request verification' if cfg['person_inside'] else 'not configured',
                safety_rule='Retain existing rule alarm. Missing information is not evidence of safety.'))
        if self.args.score:
            state['candidate_action']=dict(id='VERIFY_CURRENT_VIEW',
                description='Ask the operator to review this camera and its ROI boundary now.')
        return state

    def step_all(self):
        """Gradio가 0.2초마다 부르는 화면 갱신. 최신 프레임 위에 ROI, 박스, 이탈 배너를 그린다."""
        import cv2
        import numpy as np
        out=[]
        with self.lock:
            snapshots=[(r['frame'],deepcopy(r['cfg']),r['rendered'],r['session'],
                        r['banner_text'],r['banner_until']) for r in self.runtime.values()]
        for frame,cfg,rendered,session,banner_text,banner_until in snapshots:
            if frame is None:out.append(None);continue
            canvas=frame.copy();h,w=canvas.shape[:2]
            if cfg and cfg['polygon']:
                pts=(np.asarray(cfg['polygon'])*[w-1,h-1]).astype(np.int32)
                cv2.polylines(canvas,[pts],True,(0,220,230),2)
            cv2.putText(canvas,'ROI UNSET' if not cfg or not cfg['polygon'] else
                        'ARMED' if cfg['enabled'] else 'DISARMED',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,230),2)
            # 분석이 느려도 마지막 탐지 박스는 세션이 같을 때까지 계속 그린다.
            if rendered and rendered[1]==session:
                for d in rendered[2]:
                    x1,y1,x2,y2=d['bbox'];color=(0,0,255) if d['danger'] else (0,210,0)
                    zone=d.get('zone') or ''
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    cv2.putText(canvas,f"{d['name']} {d['confidence']:.2f} #{d['track_id']} {zone}",(x1,max(18,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
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
        lines=['### 카메라 상태',f'진행 중 규칙 경고: **{engine.db.active_count()}건**']
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
                    f"[{last.get('detected_at','')}] {last.get('camera_id','')} · {last.get('object','')} "
                    f"{last.get('event_type','')}</h3><p>{last.get('message','')}</p></div>")
        return ('<div class="alert-panel"><h3>진행 중 경고 없음 — 영상·분석 연결 상태를 함께 확인하세요.</h3></div>')
    original.alert_banner=alert_banner

    def roi_load(label):
        """ROI 편집기에 현재 다각형을 'x,y; x,y' 원본 픽셀 좌표로 채워 준다."""
        cam=next(c for c in original.CAMERAS if c['code']==label);r=engine.runtime[cam['cam_id']]
        with engine.lock:
            cfg=r['cfg'];size=r['native_size']
            text='; '.join(f'{round(x*(size[0]-1))},{round(y*(size[1]-1))}' for x,y in cfg['polygon']) if cfg and size else ''
            return text,bool(cfg and cfg['enabled']),bool(cfg and cfg['person_inside']),bool(cfg and cfg['reviewed'])

    def roi_save(label,text,enabled,person_inside,reviewed):
        """편집한 ROI를 DB에 저장하고 그 카메라 추적을 리셋한다. 교차하는 다각형은 거절한다."""
        cam=next(c for c in original.CAMERAS if c['code']==label);r=engine.runtime[cam['cam_id']]
        try:
            with engine.lock:
                if not r['native_size']:raise ValueError('영상이 준비되지 않았습니다.')
                w,h=r['native_size'];pts=original.text_to_zone(text) if text.strip() else np.empty((0,2))
                if enabled and len(pts)<3:raise ValueError('ROI를 먼저 설정하세요.')
                if len(pts) and (np.any(pts<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]>=h)):
                    raise ValueError(f'원본 좌표 범위: x=0~{w-1}, y=0~{h-1}')
                if len(pts) and (abs(cv2.contourArea(pts.astype(np.float32)))<10 or not cv2.isContourConvex(pts.astype(np.int32))):
                    raise ValueError('이 편집기는 교차하지 않는 볼록 다각형을 지원합니다.')
                version=r['roi_version']+1
                cfg=dict(polygon=(pts/[w-1,h-1]).tolist(),enabled=bool(enabled),
                         person_inside=bool(person_inside),reviewed=bool(reviewed),version=version)
                with engine.db.connect() as db:
                    db.execute('INSERT OR REPLACE INTO roi VALUES(?,?)',(cam['cam_id'],json.dumps(cfg)))
                engine.reset(r,'roi_changed');r['cfg']=cfg;r['roi_version']=version
            return 'ROI 저장 완료. 이전 JEV 판단은 새 구역에 재사용하지 않습니다.'
        except (ValueError,TypeError) as exc:return str(exc)

    def jev_view():
        """JEV 탭: 최근 요청 표, 마지막에 보낸 JSON, 실제로 받은 응답. 응답을 지어내지 않는다."""
        records=engine.db.latest();rows=[];total_in=total_out=0
        for item in records:
            state=json.loads(item['state_json']);response=json.loads(item['result_json']) if item['result_json'] else {}
            answers=response.get('answers',{});usage=response.get('usage',{})
            total_in+=usage.get('input_tokens',0);total_out+=usage.get('output_tokens',0)
            choice=answers.get('situation',{});noul=answers.get('needs_immediate_review',{});score=answers.get('action_suitability',{})
            rows.append([item['requested_at'][11:19],state['source']['camera_id'],item['status'],
                '진행 중' if item['event_active'] else '종료됨',choice.get('choice','-'),
                choice.get('confidence'),noul.get('noul'),score.get('score'),usage.get('input_tokens',0),item['error'] or ''])
        mode=('Ollama 로컬 판단 · '+engine.args.jev_model if getattr(engine.args,'provider','')=='ollama' else
              'TypeSafe API 사용') if engine.args.jev else 'JSON 준비 모드 — 외부 전송 없음'
        summary=f'### {mode}\n최근 50건 입력 토큰: {total_in} / 출력 토큰: {total_out}\n\nJEV는 참고 판단입니다. 기존 규칙 경고를 해제하거나 설비를 제어하지 않습니다.'
        newest=records[0] if records else None
        request={'model':engine.args.jev_model,'state':json.loads(newest['state_json']),
                 'questions':questions(engine.args.score)} if newest else {}
        response=json.loads(newest['result_json']) if newest and newest['result_json'] else {}
        return summary,rows,request,response

    # 화면: 실시간 4채널, 통계, 사건 표는 원본 레이아웃. JEV 판단 탭만 여기 추가한다.
    with gr.Blocks(title='SafeZoo · YOLO + JEV') as demo:
        gr.HTML(original.header_html());gr.HTML(original.hero_html())
        gr.Markdown('영상 재생·YOLO 분석은 백그라운드에서 동작합니다. JEV에는 사건 요약 JSON만 전송합니다.')
        banner=gr.HTML()
        with gr.Tabs():
            with gr.Tab('실시간 감시'):
                images=[]
                for pair in (original.CAMERAS[:2],original.CAMERAS[2:]):
                    with gr.Row():
                        for cam in pair:
                            images.append(gr.Image(label=cam['title'],height=280,interactive=False,format='jpeg'))
                with gr.Accordion('ROI 설정 — 원본 영상 픽셀 좌표',open=False):
                    camera=gr.Dropdown([c['code'] for c in original.CAMERAS],value='CAMERA 02',label='카메라')
                    get=gr.Button('현재 설정 읽기');coords=gr.Textbox(label='x,y; x,y; ...',lines=3)
                    enabled=gr.Checkbox(label='구역 감시 활성화');person=gr.Checkbox(label='사람이 ROI 안에 있으면 경고')
                    reviewed=gr.Checkbox(label='운영자가 ROI 경계를 검토함')
                    save=gr.Button('저장');message=gr.Markdown()
                    get.click(roi_load,camera,[coords,enabled,person,reviewed])
                    camera.change(roi_load,camera,[coords,enabled,person,reviewed])
                    save.click(roi_save,[camera,coords,enabled,person,reviewed],message)
                cam_md=gr.Markdown();live_kpi=gr.Markdown();inbox=gr.Textbox(label='오늘 기본 알림',lines=5)
                gallery=gr.Gallery(label='오늘 스냅샷',columns=3,height=230)
            kpi,cam_stat,chart_type,chart_sev=original.build_stats_tab()
            kw,button,table,inbox_full=original.build_events_tab()
            with gr.Tab('JEV 판단'):
                summary=gr.Markdown()
                decision_table=gr.Dataframe(headers=['시각','카메라','요청 상태','사건 상태','Choice','Choice confidence',
                    'Noul P(yes)','Score 0~3','입력 토큰','오류'],interactive=False)
                gr.Markdown('disabled: 전송 안 함 · completed: 응답 시점에 유효 · stale: 지난 상태의 응답 · error: 기본 경고 유지. '
                            '표범 cat/dog/bear와 판다 bear는 대체 클래스이며 실제 종 확인이 아닙니다.')
                request_json=gr.JSON(label='최근 JEV 요청: State + Questions')
                response_json=gr.JSON(label='최근 실제 API 응답 (가짜 응답 생성 안 함)')
        outputs=[banner,live_kpi,kpi,cam_md,cam_stat,chart_type,chart_sev,table,inbox,inbox_full,gallery]
        demo.load(original.refresh_all,kw,outputs)
        button.click(original.refresh_all,kw,outputs);kw.change(original.refresh_all,kw,outputs)
        gr.Timer(2).tick(original.refresh_all,kw,outputs)
        # 영상 칸만 0.2초마다 다시 그린다. 표와 배너는 2초마다 갱신한다.
        demo.load(engine.step_all,None,images)
        gr.Timer(.2).tick(engine.step_all,None,images,queue=False)
        demo.load(jev_view,None,[summary,decision_table,request_json,response_json])
        gr.Timer(2).tick(jev_view,None,[summary,decision_table,request_json,response_json])
    return demo


def parse_args():
    """실행 옵션. 신뢰도 기본값은 0.1이라 점수가 낮은 기린·판다 박스도 남긴다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path(r'C:\code\code_test\zoo'))  # 영상·zoo_vision.py 위치
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'zoo_jev_data')  # SQLite와 스냅샷
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


def main():
    """옵션 해석 → 원본 대시보드 로드 → 분석 시작 → Gradio 서버. 종료 시 스레드를 정리한다."""
    args=parse_args();original=load_original(args.source_dir)
    engine=LinkedAnalyzer(original,args)
    demo=build_app(original,engine)
    engine.start()
    print(f'SafeZoo: http://127.0.0.1:{args.port} | decision: {args.provider if args.jev else "JSON only"} / {args.jev_model}')
    try:
        demo.launch(server_name='127.0.0.1',server_port=args.port,share=args.share,
                    auth=('zoo',os.environ['SAFEZOO_PASSWORD']) if args.share else None,
                    theme=original.build_theme(),css=original.load_css(),
                    allowed_paths=[str(engine.db.root/'snapshots')],
                    blocked_paths=[str(engine.db.path)])
    finally:
        engine.close()


if __name__=='__main__':
    main()  # python zoo_vision_jev.py 로 직접 실행할 때만 서버를 연다.
