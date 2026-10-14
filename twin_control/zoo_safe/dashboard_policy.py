"""탐지 후보와 경고 확정을 나눈다.

신뢰도·연속 관측으로 객체를 확인하고, 그 다음 구역 전환을 본다.
완충 폭은 프레임 짧은 변 길이의 비율이다. 해상도가 바뀌어도 같은 비율을 쓴다.
숫자는 시연용이며 현장 검증값이 아니다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoGate:
    """카메라·클래스에 저장하는 시연용 기준."""
    candidate_min: float = 0.10
    confirm_min: float = 0.25
    min_samples: int = 2
    min_span_sec: float = 1.0
    max_gap_sec: float = 3.0
    boundary_norm: float = 0.02
    safe_return_sec: float = 0.8

    def as_dict(self):
        return dict(candidate_min=self.candidate_min, confirm_min=self.confirm_min,
                    min_samples=self.min_samples, min_span_sec=self.min_span_sec,
                    max_gap_sec=self.max_gap_sec, boundary_norm=self.boundary_norm,
                    safe_return_sec=self.safe_return_sec,
                    unit='candidate_min/confirm_min=confidence 0-1; time=seconds; '
                         'boundary_norm=fraction of min(frame width, height)')


def gate_from_values(values, fallback):
    """저장값 중 알려진 항목만 반영한다."""
    base = fallback.as_dict()
    for key in ('candidate_min', 'confirm_min', 'min_samples', 'min_span_sec',
                'max_gap_sec', 'boundary_norm', 'safe_return_sec'):
        if key in values and values[key] is not None:
            base[key] = values[key]
    base.pop('unit', None)
    return DemoGate(**base)


def zone_relation(signed_distance, margin_px, at_bottom, animal_bottom_is_risk):
    """발 기준점과 ROI 경계의 관계. 경계 띠 안은 안/밖으로 확정하지 않는다."""
    if animal_bottom_is_risk and at_bottom:
        return 'bottom_edge'
    if signed_distance is None:
        return 'unknown'
    if abs(float(signed_distance)) <= float(margin_px):
        return 'boundary'
    return 'inside' if signed_distance > 0 else 'outside'


def subject_side(name, relation, person_inside):
    """위험 관계와 안전 관계를 동물/사람으로 나눈다."""
    if relation in ('boundary', 'unknown', ''):
        return 'boundary' if relation == 'boundary' else 'ignore'
    if name == 'person':
        if not person_inside:
            return 'ignore'
        return 'risk' if relation == 'inside' else 'safe'
    if relation in ('outside', 'bottom_edge'):
        return 'risk'
    if relation == 'inside':
        return 'safe'
    return 'ignore'


class ObjectWatch:
    """한 추적 ID의 후보 확인과 경고 전환.

    같은 frame_key는 한 번만 반영한다. 경계 띠는 경고를 만들거나 끝내지 않는다.
    안전 쪽이 확인된 객체의 명확한 위험 전환은 바로 연다.
    처음부터 위험 쪽에서만 보인 후보는 확인 시간이 찬 뒤에만 연다.
    """

    def __init__(self):
        self.samples = []
        self.last_frame = None
        self.last_clear = None
        self.safe_since = None
        self.event_open = False
        self.confirmed = False

    def update(self, mono, frame_key, confidence, relation, gate, person_inside, name):
        if frame_key is not None and frame_key == self.last_frame:
            return {'action': 'duplicate', 'kind': None, 'confirmed': self.confirmed}
        self.last_frame = frame_key
        if confidence < gate.candidate_min:
            return {'action': 'below_candidate', 'kind': None, 'confirmed': self.confirmed}
        side = subject_side(name, relation, person_inside)
        self.samples.append((float(mono), float(confidence), side))
        del self.samples[:-64]
        self.confirmed = _confirmed(self.samples, gate)
        if side == 'boundary':
            return {'action': 'hold', 'kind': None, 'confirmed': self.confirmed}
        if side == 'ignore':
            return {'action': 'none', 'kind': None, 'confirmed': self.confirmed}
        if confidence < gate.confirm_min:
            return {'action': 'candidate', 'kind': None, 'confirmed': self.confirmed}
        if side == 'risk':
            self.safe_since = None
            kind = None
            action = 'none'
            if not self.event_open and self.confirmed and self.last_clear == 'safe':
                action, kind = 'open', 'transition'
                self.event_open = True
            elif not self.event_open and self.confirmed and _risk_span(self.samples, gate) >= gate.min_span_sec:
                action, kind = 'open', 'sustained'
                self.event_open = True
            self.last_clear = 'risk'
            return {'action': action, 'kind': kind, 'confirmed': self.confirmed}
        if self.safe_since is None:
            self.safe_since = float(mono)
        action = 'none'
        if self.event_open and float(mono) - self.safe_since >= gate.safe_return_sec:
            action = 'close'
            self.event_open = False
            self.safe_since = None
        self.last_clear = 'safe'
        return {'action': action, 'kind': None, 'confirmed': self.confirmed}

    def mark_closed(self):
        self.event_open = False
        self.safe_since = None


def _confirmed(samples, gate):
    good = [(mono, conf) for mono, conf, _side in samples if conf >= gate.confirm_min]
    if len(good) < gate.min_samples:
        return False
    start = prev = good[0][0]
    count = 1
    for mono, _conf in good[1:]:
        if mono - prev > gate.max_gap_sec:
            start, count = mono, 1
        else:
            count += 1
        prev = mono
        if count >= gate.min_samples and mono - start >= gate.min_span_sec:
            return True
    return False


def _risk_span(samples, gate):
    times = []
    for mono, conf, side in reversed(samples):
        if conf < gate.confirm_min or side == 'boundary':
            continue
        if side != 'risk':
            break
        times.append(mono)
    if len(times) < gate.min_samples:
        return 0.0
    times.reverse()
    if times[-1] - times[0] > gate.max_gap_sec * max(len(times) - 1, 1):
        return 0.0
    return times[-1] - times[0]


END_REASON_KO = {
    'safe_return': '안전 복귀',
    'track_lost_not_confirmed_safe': '추적 소실',
    'track_gap_or_class_change': '추적 단절',
    'video_loop': '영상 반복',
    'server_restart': '서버 중단',
    'server_shutdown': '서버 종료',
    'roi_changed': '구역 변경',
    'observation_lost': '영상 끊김',
    'analysis_error': '분석 오류',
}
