"""AI 판단(TEV)에 보내는 사건 요약과, 사람 판정과의 채점.

서버(twin_server.py)와 채점 도구(twin_eval.py)가 같이 쓴다. TEV에 무엇을 보내는지 바꾸려면
review_state만 고치고 twin_eval.py --ask 로 점수가 오르는지 본다.
"""
import json
import re

import twin_vision

HUMAN_LABELS = ('실제 위험', '오탐', '판단 보류')
# TEV의 상황 분류를 사람의 검토 구분에 맞춘다.
RISK_CHOICES = ('BOUNDARY_CONCERN', 'PERSON_INTRUSION_CONCERN')
CLEAR_CHOICES = ('NORMAL', 'AUTHORIZED_ACTIVITY')
ANSWERED = ('completed', 'stale', 'batch_completed', 'eval_completed')
SEVERITY_EN = {'경고': 'warning', '주의': 'caution', '참고': 'info'}
SAME_CAMERA_WINDOW_SEC = 600


def seconds(stamp):
    """'YYYY-MM-DD HH:MM:SS'의 하루 중 초. 형식이 다르면 0."""
    match = re.search(r'(\d\d):(\d\d):(\d\d)', stamp or '')
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) if match else 0


def predicted_label(choice):
    if choice in RISK_CHOICES:
        return '실제 위험'
    if choice in CLEAR_CHOICES:
        return '오탐'
    return '판단 보류'


def review_state(db, zj, row, day_rows):
    """끝난 사건을 나중에 판단시킬 때 보내는 요약. 사건 당시 엔진이 저장해 둔 요약이 있으면 그것을 쓴다.

    db는 열려 있는 sqlite 연결, row와 day_rows는 사건 dict (그날의 다른 사건 포함).
    """
    stored = db.execute("SELECT state_json FROM decisions WHERE event_id=? AND state_json IS NOT NULL "
                        "AND status NOT LIKE 'batch%' AND status NOT LIKE 'eval%' "
                        'ORDER BY rowid DESC LIMIT 1', (row['id'],)).fetchone()
    cid, name = row.get('camera_id'), row.get('object')
    if stored:
        state = json.loads(stored[0])
    else:
        person = row.get('event_type') == 'PERSON_INTRUSION'
        state = dict(
            schema_version='1.0',
            source=dict(camera_id=cid, mode='recorded_video_demo', observed_at=row.get('detected_at')),
            subject=dict(detected_class=name, class_is_proxy=twin_vision.is_proxy(cid, name),
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
    window = [other for other in day_rows
              if other['id'] != row['id'] and other.get('camera_id') == cid
              and (other.get('detected_at') or '') <= at
              and seconds(at) - seconds(other.get('detected_at') or '') <= SAME_CAMERA_WINDOW_SEC]
    state['review'] = dict(
        mode='post_hoc_review', event_still_active=bool(row['active']), end_reason=row['end_reason'],
        rule_event_type=row.get('event_type'), rule_severity=SEVERITY_EN.get(row.get('severity')),
        same_camera_events_previous_10min=len(window))
    return state


def answer_of(result):
    """TEV 응답에서 채점에 쓰는 값만 꺼낸다."""
    situation = result['answers']['situation']
    return dict(choice=situation['choice'],
                probability=situation['probabilities'][situation['choice']],
                needs_review=result['answers']['needs_immediate_review']['noul'])


def score(pairs, auto_probability=0.6, auto_needs_review=0.3):
    """pairs는 (사람 판정, answer_of 결과)의 목록.

    돌려주는 값: 사람×AI 표, 일치율, 실제 위험을 AI가 위험으로 본 비율, 실제 위험을 오탐으로 본 건수,
    오탐을 AI도 오탐으로 본 비율, 사람 판정별 '즉시 확인 필요도' 평균, 자동 확인 조건을 적용했을 때의 결과.
    """
    table = {human: {ai: 0 for ai in HUMAN_LABELS} for human in HUMAN_LABELS}
    need = {human: [] for human in HUMAN_LABELS}
    auto = dict(count=0, wrong=0)
    for human, answer in pairs:
        ai = predicted_label(answer['choice'])
        table[human][ai] += 1
        need[human].append(answer['needs_review'])
        if (ai == '오탐' and answer['probability'] >= auto_probability
                and answer['needs_review'] <= auto_needs_review):
            auto['count'] += 1
            auto['wrong'] += human == '실제 위험'
    total = len(pairs)
    danger, false = sum(table['실제 위험'].values()), sum(table['오탐'].values())
    return dict(
        total=total, table=table,
        agreement=sum(table[label][label] for label in HUMAN_LABELS) / total if total else None,
        danger_total=danger,
        danger_caught=table['실제 위험']['실제 위험'] / danger if danger else None,
        danger_missed=table['실제 위험']['오탐'],
        false_total=false,
        false_cleared=table['오탐']['오탐'] / false if false else None,
        needs_review_mean={human: sum(values) / len(values) if values else None
                           for human, values in need.items()},
        auto=dict(auto, probability=auto_probability, needs_review=auto_needs_review))


# ---------- R6: 실시간 판단 요청의 실패 원인 구분, 자동 재시도, 늦은 답 붙이기 ----------

BACKOFF_SEC = (5, 10, 20, 40, 60)   # 연속 실패 횟수에 따라 다음 시도까지 기다리는 시간
LATENCY_KEEP = 30                    # 응답 시간 통계에 쓰는 최근 성공 수


def describe_error(exc, timeout=None):
    """Ollama 호출 예외를 (종류, 사람이 읽을 원인, 같은 요청을 다시 보낼지)로 바꾼다.
    응답 본문은 Ollama가 돌려준 오류 문장만 짧게 남긴다."""
    import socket
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        detail = ''
        try:
            detail = str(json.loads(exc.read(2000) or b'{}').get('error', ''))[:160]
        except (ValueError, OSError, AttributeError):
            pass
        if 'no longer supported' in detail:
            return ('model_format', 'Ollama가 이 모델 형식을 더 이상 지원하지 않음 (ollama pull로 모델을 다시 받아야 함)', True)
        if exc.code == 404 or 'not found' in detail:
            return ('model_missing', 'Ollama에 모델이 없음 (ollama pull로 받아야 함)', True)
        if exc.code >= 500:
            return ('ollama_error', f'Ollama 내부 오류 HTTP {exc.code}' + (f': {detail}' if detail else ''), True)
        return ('http', f'HTTP {exc.code}' + (f': {detail}' if detail else ''), True)
    reason = getattr(exc, 'reason', exc)
    if isinstance(reason, ConnectionRefusedError) or isinstance(exc, ConnectionRefusedError):
        return ('ollama_off', 'Ollama가 꺼져 있음 (연결 거부)', True)
    if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(exc, (socket.timeout, TimeoutError)):
        return ('timeout', f'응답 시간 초과 ({timeout:g}초)' if timeout else '응답 시간 초과', True)
    if isinstance(exc, urllib.error.URLError):
        return ('network', f'Ollama에 연결하지 못함 ({type(reason).__name__})', True)
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        # 모델이 답은 했지만 형식이 틀렸다. 같은 요청을 다시 보내도 대개 같으므로 다시 보내지 않는다.
        return ('bad_answer', f'모델 응답 형식 오류 ({type(exc).__name__})', False)
    return ('other', type(exc).__name__, True)


def make_worker(zj):
    """엔진의 JevWorker를 바꾼 판단 요청 담당. 엔진 파일은 고치지 않는다.

    엔진 기본 동작과 다른 점:
    - 실패하면 원인을 구분해 기록하고, 요청을 버리지 않고 대기열 앞에 다시 넣어 5~60초 간격으로 재시도한다
      (엔진은 30초 동안 들어오는 요청을 모두 'API cooldown'으로 버렸다).
    - 사건이 끝났거나 오래된 요청도 보낸다. 늦게 온 답은 'stale'(사건 종료 후 답)로 그 사건에 붙는다
      (엔진은 보내기 전에 버렸다). 대기열 상한 32건과 같은 사건의 새 요청이 옛 요청을 대신하는 규칙은 그대로다.
    - 상태(정상/대기/연결 중/오류와 원인), 마지막 성공 시각, 응답 시간을 화면에 넘긴다.
    """
    import collections
    import copy
    import statistics
    import time
    import urllib.request

    class TwinJevWorker(zj.JevWorker):
        def __init__(self, engine, transport=None):
            super().__init__(engine, transport)
            self.status = 'connecting'
            self.reason = ''
            self.kind = ''
            self.failures = 0          # 연속 실패 수
            self.total_failures = 0
            self.next_try = 0.0
            self.last_ok = None
            self.last_error_at = None
            self.late = 0              # 사건이 끝난 뒤 붙은 답 수
            self.latencies = collections.deque(maxlen=LATENCY_KEEP)
            self.last_state = None     # 점검용으로 다시 보낼 최근 사건 요약
            self.last_check = 0.0

        def retry_now(self):
            """화면의 '다시 시도' 버튼. 기다리던 간격을 무시하고 바로 다음 요청(없으면 점검)을 보낸다."""
            with self.condition:
                self.next_try = 0.0
                self.last_check = 0.0
                if self.status == 'down':
                    self.status = 'connecting'
                self.condition.notify()

        def snapshot(self):
            with self.condition:
                lat = sorted(self.latencies)
                return dict(status=self.status, kind=self.kind, reason=self.reason, failures=self.failures,
                            total_failures=self.total_failures, queue=len(self.pending),
                            retry_in=max(0, round(self.next_try - time.monotonic())) if self.status == 'down' else 0,
                            last_ok=self.last_ok, last_error_at=self.last_error_at, late_answers=self.late,
                            latency_ms=round(statistics.median(lat)) if lat else None,
                            latency_max_ms=lat[-1] if lat else None, latency_n=len(lat))

        def failed(self, exc):
            kind, reason, again = describe_error(exc, self.engine.args.jev_timeout)
            with self.condition:
                self.failures += 1
                self.total_failures += 1
                self.status, self.kind, self.reason = 'down', kind, reason
                self.last_error_at = zj.now().isoformat(timespec='seconds')
                self.next_try = time.monotonic() + BACKOFF_SEC[min(self.failures, len(BACKOFF_SEC)) - 1]
            return reason, again

        def succeeded(self, latency_ms):
            with self.condition:
                self.status, self.kind, self.reason, self.failures, self.next_try = 'ok', '', '', 0, 0.0
                self.last_ok = zj.now().isoformat(timespec='seconds')
                self.latencies.append(latency_ms)

        def sample_state(self):
            if self.last_state is None:
                with self.engine.db.connect() as db:
                    row = db.execute('SELECT state_json FROM decisions WHERE state_json IS NOT NULL '
                                     'ORDER BY rowid DESC LIMIT 1').fetchone()
                self.last_state = json.loads(row[0]) if row else None
            return self.last_state

        def probe(self):
            """대기열이 비었을 때의 점검. 오류 중이면 최근 사건 요약으로 실제 판단을 보내 보고(기록은 남기지 않음),
            정상일 때는 1분마다 모델 목록만 가볍게 확인한다."""
            args = self.engine.args
            self.last_check = time.monotonic()
            try:
                if self.status == 'ok':
                    with urllib.request.urlopen(args.ollama_url.rstrip('/') + '/api/tags', timeout=3) as r:
                        names = [m.get('name') for m in json.loads(r.read(200_000)).get('models', [])]
                    if args.jev_model not in names:
                        raise urllib.error.HTTPError(args.ollama_url, 404, 'not found', None, None)
                    return
                state = self.sample_state()
                if state is None:
                    return
                started = time.monotonic()
                zj.validate_answer(self.transport(copy.deepcopy(state), args), args.score)
                self.succeeded(round((time.monotonic() - started) * 1000))
            except Exception as exc:
                self.failed(exc)

        def run(self):
            engine = self.engine
            while not engine.stop.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda: engine.stop.is_set() or (self.pending and time.monotonic() >= self.next_try),
                        timeout=.5)
                    if engine.stop.is_set():
                        return
                    now = time.monotonic()
                    idle_check = not self.pending and (
                        (self.status != 'ok' and now >= self.next_try and now - self.last_check >= 5)
                        or (self.status == 'ok' and now - self.last_check >= 60))
                    job = None
                    if self.pending and now >= self.next_try:
                        key, job = self.pending.popitem(last=False)
                if job is None:
                    if idle_check:
                        self.probe()
                    continue
                self.last_state = job['state']
                engine.db.decision(job['id'], 'running')
                started = time.monotonic()
                try:
                    response = zj.validate_answer(self.transport(copy.deepcopy(job['state']), engine.args), engine.args.score)
                except Exception as exc:
                    reason, again = self.failed(exc)
                    with self.condition:
                        requeue = again and key not in self.pending
                        if requeue:
                            # 같은 사건의 더 새 요청이 없으면 앞에 다시 넣는다. 사건 쪽 inflight는 그대로 둔다.
                            self.pending[key] = job
                            self.pending.move_to_end(key, last=False)
                    if requeue:
                        engine.db.decision(job['id'], 'pending', error=reason)
                    else:
                        engine.db.decision(job['id'], 'error', error=reason)
                        engine.finish_job(job, 'error')
                    continue
                response['latency_ms'] = round((time.monotonic() - started) * 1000)
                with engine.lock:
                    status = 'completed' if self.current(job) else 'stale'
                    engine.db.decision(job['id'], status, response)
                engine.finish_job(job, status)
                self.succeeded(response['latency_ms'])
                with self.condition:
                    self.late += status == 'stale'

    return TwinJevWorker
