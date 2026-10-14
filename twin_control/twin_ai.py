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
