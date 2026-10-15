"""카메라 구성과 탐지 후처리.

분석 엔진(zoo_safe)은 고치지 않는다. 엔진이 부르는 YOLO 모델을 DetectorProxy로 감싸서,
엔진에 넘어가기 전에 탐지 결과를 정리한다.

    1. 종 묶기      기본 YOLO에 없는 종은 여러 클래스로 번갈아 잡힌다 (호랑이 → zebra, giraffe, cow).
                    카메라마다 정한 클래스들을 그 카메라의 종 하나로 묶는다.
    2. 겹침 제거    같은 개체가 두 클래스로 동시에 잡혀 생긴 겹친 박스를 하나만 남긴다.
    3. 정지 물체    같은 자리에 오래 있으면서 신뢰도가 낮은 박스는 바위 같은 배경으로 보고 뺀다.
    4. 최소 신뢰도  카메라별 기준 미만은 버린다.

아래 숫자는 2026-10-13의 시험 영상 4개를 재서 정한 값이다. 현장 카메라에서는 다시 재야 한다.
묶기는 이름만 바꾼다. 줄무늬 옷이나 다른 동물을 호랑이와 구별해 주지는 못한다.
"""
from copy import deepcopy
import json
from dataclasses import replace
import statistics
import time
import types

PERSON = 'person'
PERSON_MIN = 0.40      # 사람 박스 최소 신뢰도
DUP_IOU = 0.55         # 같은 종 박스가 이만큼 겹치면 같은 개체로 본다
ANCHOR_IOU = 0.50      # 정지 물체 자리와 이만큼 겹치면 같은 자리로 본다
STATIC_SEC = 8.0       # 이 시간 넘게 같은 자리에 있고
STATIC_SAMPLES = 12    # 이만큼 넘게 관측됐고
STATIC_CONF = 0.50     # 신뢰도 중앙값이 이보다 낮으면 정지 물체다
STATIC_PASS = 0.65     # 정지 물체 자리라도 이 신뢰도 이상인 박스는 살린다 (그 앞에 선 동물)
ANCHOR_WINDOW = 60.0   # 자리마다 최근 이 시간의 관측만 본다
ANCHOR_FORGET = 30.0   # 이 시간 동안 안 보인 자리는 잊는다
ALIAS_BASE = 1000      # 엔진에 알려 줄 가짜 클래스 번호의 시작
# 이탈 확정 (R1, 2026-10-15): 안전 쪽에 있던 동물이 밖으로 나간 것으로 보여도, 밖에서 연속으로 이만큼
# 관측돼야 알림을 연다. 엔진 기본 규칙은 밖으로 나간 첫 관측(초당 약 3장 중 한 장)에서 바로 알림을 열었다.
EXIT_SAMPLES = 3
EXIT_SEC = 0.6

# species: 엔진과 사건 기록에 남는 이름.  label: 화면에 쓰는 이름.
# classes: 그 종으로 묶을 YOLO 클래스.  proxy: 묶은 클래스가 실제 종명이 아니다.
# static_filter: 정지 물체 제외를 쓴다. 가만히 있는 동물이 낮은 신뢰도로 잡히는 카메라에서는 끈다.
CAMERAS = (
    dict(cam_id='ZOO_CAM_ELEPHANT_01', code='CAMERA 01', title='코끼리 방사장', desc='대형 동물 경계 · 이동 감시',
         file='elephant_cctv.mp4', species='elephant', label='코끼리', classes=('elephant',),
         min_conf=0.45, proxy=False, static_filter=True, exit_samples=EXIT_SAMPLES, exit_sec=EXIT_SEC, boundary_norm=None),
    dict(cam_id='ZOO_CAM_GIRAFFE_01', code='CAMERA 02', title='기린 관람 구역', desc='관람객 안전선 · 구역 감시',
         file='giraffe_cctv.mp4', species='giraffe', label='기린', classes=('giraffe',),
         min_conf=0.25, proxy=False, static_filter=False,
         # R3 (2026-10-15): 서 있는 기린이 사람으로 잡혔다 (신뢰도 최대 0.67). 사람 기준을 0.70으로 올리고,
         # 최근 2초 안에 기린이 있던 자리와 겹치는 사람 박스는 기린으로 본다.
         person_min=0.70, species_hold=dict(sec=2.0, iou=0.3),
         # 뛰는 기린은 초당 3장 사이에 박스가 겹치지 않아 추적이 새로 시작됐다. 가까운 곳에서 1초 안에 다시 잡히면 잇는다.
         track_jump=dict(missed=3, scale=1.5)),
    dict(cam_id='ZOO_CAM_TIGER_01', code='CAMERA 03', title='호랑이 방사장', desc='맹수 경계 · 이동 감시',
         file='tiger_cctv.mp4', species='tiger', label='호랑이 추정',
         classes=('zebra', 'giraffe', 'cow', 'sheep', 'horse', 'cat', 'dog', 'bear'),
         min_conf=0.25, proxy=True, static_filter=False),
    dict(cam_id='ZOO_CAM_REDPANDA_01', code='CAMERA 04', title='레서판다 관람 구역', desc='관람객·동물 접근 감시',
         file='redpanda_cctv.mp4', species='red_panda', label='레서판다 추정',
         classes=('dog', 'cat', 'bear', 'cow', 'sheep', 'teddy bear'),
         min_conf=0.25, proxy=True, static_filter=False,
         # R2 (2026-10-15): 레서판다가 사람으로 잡힌 박스는 신뢰도가 최대 0.62였다. 사람 기준을 0.65로 올린다.
         person_min=0.65,
         # 사람 침입은 빨리 알린다: 경계 띠 0.5%(2.4px), 확인 시간 0.5초 (엔진 기본 2%, 1초)
         person_gate=dict(boundary_norm=0.005, min_span_sec=0.5),
         # 고객 요청 (R2 후보 검토 중 추가): 사람은 박스 가운데로 판정한다. 울타리를 넘는 순간을 발보다 먼저 잡는다.
         person_anchor='center'),
)
BY_ID = {cam['cam_id']: cam for cam in CAMERAS}
LABELS = dict({cam['species']: cam['label'] for cam in CAMERAS}, person='사람')
ZONE_KO = {'SAFE': '안전', 'DANGER': '위험', 'BOUNDARY': '경계'}


def camera_gate(cam_id, name, gate):
    """카메라별 판정 기준. 동물은 boundary_norm(경계 완충 폭), 사람은 person_gate(완충 폭, 확인 시간 등).
    값이 없으면 엔진 기본값 그대로."""
    cam = BY_ID.get(cam_id)
    if not cam:
        return gate
    if name == PERSON:
        return replace(gate, **cam['person_gate']) if cam.get('person_gate') else gate
    if cam.get('boundary_norm') is None:
        return gate
    return replace(gate, boundary_norm=cam['boundary_norm'])


_WATCH = {}


def object_watch_class():
    """엔진의 ObjectWatch에 이탈 확정 조건을 더한 것. 엔진 모듈 경로가 잡힌 뒤에 만든다."""
    if 'cls' in _WATCH:
        return _WATCH['cls']
    from dashboard_policy import ObjectWatch

    class TwinObjectWatch(ObjectWatch):
        """엔진은 안전 쪽에서 확인된 동물이 밖에서 한 번 관측되면 바로 알림을 연다 ('transition').
        카메라에 exit_samples가 있으면, 밖(위험 쪽, 확정 신뢰도 이상) 관측이 끊김 없이 그만큼,
        exit_sec 이상 이어진 뒤에만 연다. 경계 띠 관측이나 안전 관측이 끼면 처음부터 다시 센다."""
        camera = None      # 분석 스레드가 판정 직전에 적는다

        def update(self, mono, frame_key, confidence, relation, gate, person_inside, name):
            result = super().update(mono, frame_key, confidence, relation, gate, person_inside, name)
            cam = BY_ID.get(TwinObjectWatch.camera) or {}
            need = cam.get('exit_samples')
            if result['action'] == 'none' and self.last_clear == 'safe':
                self.held_transition = False
            if not need or name == PERSON:
                return result
            if result['action'] == 'open':
                streak = []
                for t, conf, side in reversed(self.samples):
                    if side != 'risk' or conf < gate.confirm_min:
                        break
                    streak.append(t)
                if len(streak) < need or (streak[0] - streak[-1]) < cam.get('exit_sec', 0):
                    # 아직 짧다: 연 알림을 거둬들이고, 나중에 열 때 원래 종류를 쓰도록 기억한다
                    self.event_open = False
                    if result['kind'] == 'transition':
                        self.held_transition = True
                    return dict(result, action='hold_exit', kind=None)
                if getattr(self, 'held_transition', False):
                    self.held_transition = False
                    return dict(result, kind='transition')
            return result

    _WATCH['cls'] = TwinObjectWatch
    return TwinObjectWatch


def anchor_point(cam, name, bbox, h):
    """구역 판정에 쓰는 점. 기본은 발 위치(박스 아래쪽 가운데). 카메라에 person_anchor='center'가 있으면
    사람만 박스 가운데를 쓴다 (울타리를 넘어 들어오는 사람을 발보다 먼저 잡기 위해)."""
    x1, y1, x2, y2 = bbox
    # 화면 아래 끝에 잘린 박스는 앞쪽(울타리 밖) 관람객이라 가운데가 위로 올라와 보인다: 발 위치로 판정한다
    if name == PERSON and (cam or {}).get('person_anchor') == 'center' and y2 < h - 3:
        return (x1 + x2) / 2, (y1 + y2) / 2
    return (x1 + x2) / 2, min(float(y2), h - 1)


_TRACKERS = {}


def make_tracker(cam, zj):
    """카메라에 맞는 추적기. track_jump가 있으면 겹치지 않아도 가까이 다시 잡힌 박스를 같은 추적으로 잇는다."""
    jump = (cam or {}).get('track_jump')
    if not jump:
        return zj.IoUTracker()
    if zj not in _TRACKERS:
        class JumpTracker(zj.IoUTracker):
            def __init__(self, missed, scale):
                super().__init__()
                self.jump_missed, self.jump_scale = missed, scale

            def update(self, detections):
                before = {tid: dict(t) for tid, t in self.tracks.items()}
                first_new = self.next_id
                out = super().update(detections)
                pairs = []
                for det in out:
                    if det['track_id'] < first_new:
                        continue
                    cx, cy = (det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2
                    for tid, old in before.items():
                        now = self.tracks.get(tid)
                        if old['name'] != det['name'] or not now or now['missed'] == 0 or now['missed'] > self.jump_missed:
                            continue
                        b = old['bbox']
                        dist = ((cx - (b[0] + b[2]) / 2) ** 2 + (cy - (b[1] + b[3]) / 2) ** 2) ** 0.5
                        if dist <= self.jump_scale * max(b[2] - b[0], b[3] - b[1]):
                            pairs.append((dist, tid, det))
                used = set()
                for dist, tid, det in sorted(pairs, key=lambda p: p[0]):
                    if tid in used or det['track_id'] in used:
                        continue
                    used.update((tid, det['track_id']))
                    del self.tracks[det['track_id']]
                    self.tracks[tid].update(bbox=det['bbox'], missed=0)
                    det['track_id'] = tid
                return out
        _TRACKERS[zj] = JumpTracker
    return _TRACKERS[zj](**jump)


def label_of(name):
    return LABELS.get(name, name)


def is_proxy(cam_id, name):
    cam = BY_ID.get(cam_id)
    return bool(cam and cam['proxy'] and name != PERSON)


def engine_cameras(original, source_dir):
    """엔진이 읽는 형식의 카메라 목록. 원본 zoo_vision.py에 같은 카메라가 있으면 그 구역 설정을 이어받는다."""
    known = {cam['cam_id']: cam for cam in original.CAMERAS}
    cameras = []
    for cam in CAMERAS:
        base = known.get(cam['cam_id'], dict(safe_zone=None, alert_on_exit=False))
        cameras.append(dict(base, code=cam['code'], title=cam['title'], desc=cam['desc'],
                            file=source_dir / cam['file'], cam_id=cam['cam_id'],
                            watch=(cam['species'], PERSON)))
    return cameras


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StaticFilter:
    """카메라 한 대의 '자리' 기억. 영상이 한 바퀴 돌아 추적이 끊겨도 자리는 남는다."""

    def __init__(self):
        self.anchors = []
        self.species = []        # 최근 동물 박스 (mono, bbox). species_hold 카메라에서 사람 오판을 되돌릴 때 쓴다
        self.relabeled = 0       # 사람 → 동물로 되돌린 박스 수 (진단용)

    def observe(self, bbox, confidence, mono):
        """이 박스를 기록하고, 정지 물체로 보고 빼야 하면 True."""
        self.anchors = [a for a in self.anchors if mono - a['samples'][-1][0] <= ANCHOR_FORGET]
        anchor = max(self.anchors, key=lambda a: box_iou(a['box'], bbox), default=None)
        if anchor is None or box_iou(anchor['box'], bbox) < ANCHOR_IOU:
            self.anchors.append(dict(box=list(bbox), samples=[(mono, confidence)]))
            return False
        samples = anchor['samples']
        samples.append((mono, confidence))
        while mono - samples[0][0] > ANCHOR_WINDOW:
            samples.pop(0)
        static = (mono - samples[0][0] >= STATIC_SEC and len(samples) >= STATIC_SAMPLES
                  and statistics.median(c for _, c in samples) < STATIC_CONF)
        return static and confidence < STATIC_PASS


def post_process(cam, raw, static, mono):
    """raw는 (YOLO 클래스명, 신뢰도, [x1, y1, x2, y2])의 목록. (남길 것, 정지 물체로 뺀 박스)를 돌려준다."""
    merged = []
    hold = cam.get('species_hold')
    if hold:
        static.species = [(t, b) for t, b in static.species if mono - t <= hold['sec']]
        here = [bbox for name, confidence, bbox in raw if name in cam['classes'] and confidence >= cam['min_conf']]
        recent = [b for _, b in static.species] + here
        static.species += [(mono, b) for b in here]
    for name, confidence, bbox in raw:
        if name == PERSON and hold and any(box_iou(b, bbox) >= hold['iou'] for b in recent):
            # 같은 자리에 방금까지 동물이 있었다: 같은 개체를 한두 장 사람으로 잘못 본 것으로 본다
            static.relabeled += 1
            merged.append((cam['species'], confidence, bbox))
        elif name == PERSON:
            if confidence >= cam.get('person_min', PERSON_MIN):
                merged.append((PERSON, confidence, bbox))
        elif name in cam['classes']:
            merged.append((cam['species'], confidence, bbox))
    merged.sort(key=lambda item: -item[1])
    unique = []
    for item in merged:
        if all(kept[0] != item[0] or box_iou(kept[2], item[2]) < DUP_IOU for kept in unique):
            unique.append(item)
    kept, ignored = [], []
    for name, confidence, bbox in unique:
        if name != PERSON:
            if cam['static_filter'] and static.observe(bbox, confidence, mono):
                ignored.append(bbox)
                continue
            if confidence < cam['min_conf']:
                continue
        kept.append((name, confidence, bbox))
    return kept, ignored


class _Box:
    """엔진이 읽는 만큼만 흉내 낸 YOLO 박스: cls.item(), conf.item(), xyxy[0].tolist()."""

    def __init__(self, class_id, confidence, bbox):
        import numpy as np
        self.cls = np.float32(class_id)
        self.conf = np.float32(confidence)
        self.xyxy = np.asarray([bbox], dtype=np.float32)


class DetectorProxy:
    """엔진과 YOLO 사이에 선다. 엔진에는 카메라의 종 이름을 클래스로 가진 모델처럼 보인다."""

    def __init__(self, model):
        self.model = model
        self.camera = None           # 분석 스레드가 탐지 직전에 적는다
        self.ignored = {}            # 카메라별로 마지막에 뺀 정지 물체 박스
        self.static = {cam['cam_id']: StaticFilter() for cam in CAMERAS}
        self.names = dict(model.names)
        real = {name: index for index, name in model.names.items()}
        self.real_ids = {}
        for offset, cam in enumerate(CAMERAS):
            if cam['species'] not in real:
                self.names[ALIAS_BASE + offset] = cam['species']
            self.real_ids[cam['cam_id']] = [real[name] for name in (*cam['classes'], PERSON) if name in real]
        self.ids = {name: index for index, name in self.names.items()}

    def predict(self, frame, classes=None, **options):
        cam = BY_ID[self.camera]
        result = self.model.predict(frame, classes=self.real_ids[cam['cam_id']], **options)[0]
        raw = [(self.model.names[int(box.cls.item())], float(box.conf.item()),
                [float(v) for v in box.xyxy[0].tolist()])
               for box in (result.boxes if result.boxes is not None else [])]
        kept, ignored = post_process(cam, raw, self.static[cam['cam_id']], time.monotonic())
        self.ignored[cam['cam_id']] = ignored
        return [types.SimpleNamespace(boxes=[_Box(self.ids[name], confidence, bbox)
                                             for name, confidence, bbox in kept])]


_FONTS = {}


def _font(size):
    from PIL import ImageFont
    if size not in _FONTS:
        for path in (r'C:\Windows\Fonts\malgunbd.ttf', r'C:\Windows\Fonts\malgun.ttf'):
            try:
                _FONTS[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _write(rgb, texts):
    """한글은 OpenCV 글꼴로 못 쓴다. 글자만 PIL로 얹는다. texts는 (x, y, 글, RGB 색, 크기)."""
    if not texts:
        return rgb
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for x, y, text, color, size in texts:
        draw.text((x, y), text, font=_font(size), fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(image)


def render_all(engine, box_note_sec):
    """카메라별 화면용 RGB 그림. 박스는 분석한 그 프레임 위에만 그린다 (엔진의 step_all과 같은 규칙)."""
    import cv2
    import numpy as np
    with engine.lock:
        shots = []
        for cid, r in engine.runtime.items():
            rendered = r['rendered']
            shot = None if not rendered or rendered[3] is None else rendered[3].copy()
            shots.append((cid, r['frame'], deepcopy(r['cfg']), rendered, r['session'],
                          r['banner_text'], r['banner_until'], shot, r.get('show_roi', True)))
    proxy = engine.model if isinstance(engine.model, DetectorProxy) else None
    out = []
    for cid, frame, cfg, rendered, session, banner_text, banner_until, shot, show_roi in shots:
        if frame is None and shot is None:
            out.append(None)
            continue
        age = time.monotonic() - rendered[0] if rendered else None
        fresh = bool(rendered and rendered[1] == session and shot is not None and age <= engine.args.max_gap)
        canvas = (shot if fresh else frame).copy()
        h, w = canvas.shape[:2]
        texts = []
        if show_roi and cfg and cfg.get('polygon'):
            points = (np.asarray(cfg['polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(canvas, [points], True, (0, 220, 230), 2)
            texts.append((10, 30, '구역 감시 중' if cfg.get('enabled') else '구역 감시 꺼짐', (230, 220, 0), 15))
        if show_roi and cfg and cfg.get('exclude_polygon'):
            extra = (np.asarray(cfg['exclude_polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(canvas, [extra], True, (0, 140, 255), 2)
        if fresh:
            for bbox in (proxy.ignored.get(cid, []) if proxy else []):
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (150, 150, 150), 1)
            for d in rendered[2]:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255) if d['danger'] else (0, 210, 0), 2)
                zone = ZONE_KO.get(d.get('zone') or '', '')
                texts.append((x1, max(2, y1 - 20), f"{label_of(d['name'])} {d['confidence']:.2f}"
                              + (' ' + zone if zone else ''),
                              (255, 80, 80) if d['danger'] else (60, 230, 60), 14))
            if age >= box_note_sec:
                texts.append((10, h - 26, f'이전 분석 결과 {age:.1f}초', (230, 220, 0), 15))
        elif frame is not None:
            texts.append((10, h - 26, '분석 결과 만료: 박스 숨김', (230, 220, 0), 15))
        if banner_text and time.monotonic() < banner_until:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 52), (w, 92), (0, 0, 180), -1)
            canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
            texts.append((12, 62, banner_text, (255, 255, 255), 15))
        out.append(_write(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), texts))
    return out


class LiveRenderer:
    """화면용 영상 (R5). 영상은 최신 프레임으로 부드럽게 보내고, 박스·글자는 마지막 분석 결과를 겹친다.

    분석(analyze 스레드)은 그대로 둔다. 여기서는 엔진 상태를 읽기만 한다.
    박스와 글자 층은 분석 결과·구역·배너가 바뀔 때만 다시 그리고, 매 프레임에는 겹치기만 한다.
    박스 위치는 마지막으로 분석한 프레임 기준이라, 빠르게 움직이는 대상과는 최대 분석 간격만큼 어긋날 수 있다.
    """

    def __init__(self, engine, box_note_sec):
        self.engine = engine
        self.box_note_sec = box_note_sec
        self.layers = {}     # cid -> (key, overlay BGR, mask)
        self.builds = 0      # 겹침 층을 다시 그린 횟수 (진단용)

    def frames(self, cids):
        """[(cid, BGR 그림 또는 None, 내용 키)]. 내용 키가 같으면 그림도 같다 (다시 압축할 필요 없음)."""
        import cv2
        import numpy as np
        engine = self.engine
        proxy = engine.model if isinstance(engine.model, DetectorProxy) else None
        with engine.lock:
            shots = []
            for cid in cids:
                r = engine.runtime[cid]
                shots.append((cid, r['frame'], r['seq'], r['cfg'], r['rendered'], r['session'], r['banner_text'],
                              r['banner_until'], r.get('show_roi', True),
                              tuple(map(tuple, proxy.ignored.get(cid, []))) if proxy else ()))
        out = []
        now = time.monotonic()
        for cid, frame, seq, cfg, rendered, session, banner_text, banner_until, show_roi, ignored in shots:
            if frame is None:
                out.append((cid, None, None))
                continue
            h, w = frame.shape[:2]
            age = now - rendered[0] if rendered else None
            fresh = bool(rendered and rendered[1] == session and age <= engine.args.max_gap)
            note = round(age, 1) if fresh and age >= self.box_note_sec else None
            banner = banner_text if banner_text and now < banner_until else ''
            key = (w, h, json.dumps(cfg, sort_keys=True) if show_roi and cfg else None,
                   (rendered[0], ignored) if fresh else 'stale', note, banner)
            layer = self.layers.get(cid)
            if not layer or layer[0] != key:
                overlay, mask = self.build(w, h, cfg if show_roi else None, rendered[2] if fresh else None,
                                           ignored if fresh else (), note, banner, fresh)
                layer = self.layers[cid] = (key, overlay, mask)
                self.builds += 1
            # 겹치기는 OpenCV로 한다. numpy 불리언 마스크는 카메라당 약 16ms 동안 GIL을 잡아 분석 스레드를 늦춘다.
            canvas = frame.copy()
            if banner:
                band = canvas[52:92]
                cv2.addWeighted(band, 0.25, np.full_like(band, (0, 0, 180)), 0.75, 0, dst=band)
            cv2.copyTo(layer[1], layer[2], canvas)
            out.append((cid, canvas, (seq, key)))
        return out

    def build(self, w, h, cfg, detections, ignored, note, banner, fresh):
        """박스·구역선·글자를 투명 바탕에 한 번 그린다. render_all과 같은 색과 문구."""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
        rgba = np.zeros((h, w, 4), np.uint8)
        texts = []
        if cfg and cfg.get('polygon'):
            points = (np.asarray(cfg['polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(rgba, [points], True, (230, 220, 0, 255), 2)
            texts.append((10, 30, '구역 감시 중' if cfg.get('enabled') else '구역 감시 꺼짐', (230, 220, 0), 15))
        if cfg and cfg.get('exclude_polygon'):
            extra = (np.asarray(cfg['exclude_polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(rgba, [extra], True, (255, 140, 0, 255), 2)
        if detections is not None:
            for bbox in ignored:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(rgba, (x1, y1), (x2, y2), (150, 150, 150, 255), 1)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(rgba, (x1, y1), (x2, y2), (255, 0, 0, 255) if d['danger'] else (0, 210, 0, 255), 2)
                zone = ZONE_KO.get(d.get('zone') or '', '')
                texts.append((x1, max(2, y1 - 20), f"{label_of(d['name'])} {d['confidence']:.2f}"
                              + (' ' + zone if zone else ''),
                              (255, 80, 80) if d['danger'] else (60, 230, 60), 14))
            if note is not None:
                texts.append((10, h - 26, f'이전 분석 결과 {note:.1f}초', (230, 220, 0), 15))
        elif not fresh:
            texts.append((10, h - 26, '분석 결과 만료: 박스 숨김', (230, 220, 0), 15))
        if banner:
            texts.append((12, 62, banner, (255, 255, 255), 15))
        if texts:
            image = Image.fromarray(rgba, 'RGBA')
            draw = ImageDraw.Draw(image)
            for x, y, text, color, size in texts:
                draw.text((x, y), text, font=_font(size), fill=color + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            rgba = np.asarray(image)
        return np.ascontiguousarray(rgba[..., 2::-1]), (rgba[..., 3] > 0).astype(np.uint8)


def left_safe(samples, stamp):
    """samples: [(mono, conf, side)]. 마지막 안전 관측 다음의 첫 관측이 알림 시각보다 몇 초 전인지. 안전 관측이 없으면 None."""
    last_safe = max((i for i, x in enumerate(samples) if x[2] == 'safe'), default=None)
    if last_safe is None or last_safe + 1 >= len(samples):
        return None
    return round(stamp - samples[last_safe + 1][0], 2)


def geometry_evidence(engine, cid, d, s, cfg, stamp):
    """AI에 보내는 위치 단서 (R8). 거리와 크기는 화면 크기로 나눈 비율이라 카메라 해상도와 무관하다."""
    import cv2
    frame = engine.runtime[cid]['frame']
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = d['bbox']
    out = dict(box_width_ratio=round((x2 - x1) / w, 3), box_height_ratio=round((y2 - y1) / h, 3),
               detector_confidence_now=round(float(d['confidence']), 3))
    poly = engine.pixel_polygon(cfg, w, h)
    if poly is not None and d.get('anchor'):
        fx, fy = d['anchor'][0] * (w - 1), d['anchor'][1] * (h - 1)
        signed = float(cv2.pointPolygonTest(poly, (float(fx), float(fy)), True))
        # 양수 = 구역 밖으로 나간 거리, 음수 = 구역 안쪽 깊이. 화면 짧은 변 대비.
        out['foot_outside_distance_ratio'] = round(-signed / min(w, h), 3)
    samples = s['watch'].samples if s.get('watch') else []
    if samples:
        out['track_observed_sec'] = round(stamp - samples[0][0], 2)
        out['outside_samples'] = sum(1 for x in samples if x[2] == 'risk')
    since = left_safe(samples, stamp)
    out['sec_since_last_seen_inside'] = since          # 안에서 본 적이 없으면 null
    return out


def make_analyzer(zj):
    """엔진의 LinkedAnalyzer에서 모델과 화면 그리기만 바꾼 분석기."""

    class TwinAnalyzer(zj.LinkedAnalyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 엔진의 observe()는 판정할 때마다 dashboard_policy에서 ObjectWatch를 가져온다.
            # 이탈 확정 조건을 더한 클래스로 바꿔 끼운다 (이 프로세스 안에서만).
            import dashboard_policy
            dashboard_policy.ObjectWatch = object_watch_class()
            self.exit_log = self.db.root / 'exit_diagnostics.jsonl'
            # R6: 판단 요청 담당을 바꿔 끼운다 (실패 원인 구분, 자동 재시도, 늦은 답 붙이기). 스레드는 start()에서 켜진다.
            import twin_ai
            self.worker = twin_ai.make_worker(zj)(self)

        def gate_for(self, cid, name):
            return camera_gate(cid, name, super().gate_for(cid, name))

        def observe(self, cid, detections, frame, stamp, observed_at, video_sec, cfg, frame_key=None):
            object_watch_class().camera = cid
            r = self.runtime[cid]
            before = {tid: s.get('event') for tid, s in r['tracks'].items()}
            super().observe(cid, detections, frame, stamp, observed_at, video_sec, cfg, frame_key)
            for d in detections:
                s = r['tracks'].get(d['track_id'])
                if s and s.get('event') and s['event'] != before.get(d['track_id']):
                    self.log_exit(cid, d, s, frame, stamp, video_sec, cfg)

        def log_exit(self, cid, d, s, frame, stamp, video_sec, cfg):
            """사건이 열린 순간의 근거를 한 줄씩 남긴다 (R1 개선안 I1). 실패해도 분석은 계속한다."""
            try:
                import cv2
                h, w = frame.shape[:2]
                poly = self.pixel_polygon(cfg, w, h)
                fx, fy = d['anchor'][0] * (w - 1), d['anchor'][1] * (h - 1)
                gate = self.gate_for(cid, d['name'])
                row = dict(at=zj.now().isoformat(timespec='seconds'), event=s['event'], camera=cid, track=d['track_id'],
                           object=d['name'], confidence=round(d['confidence'], 3), relation=d.get('relation'),
                           video_sec=round(video_sec, 2), bbox=d['bbox'], foot=[round(fx), round(fy)],
                           distance_px=round(float(cv2.pointPolygonTest(poly, (float(fx), float(fy)), True)), 1) if poly is not None else None,
                           margin_px=round(gate.boundary_norm * min(w, h), 1),
                           box_wh=[d['bbox'][2] - d['bbox'][0], d['bbox'][3] - d['bbox'][1]],
                           track_sec=round(stamp - s['watch'].samples[0][0], 2) if s['watch'].samples else None,
                           # R3 I5: 마지막으로 안전 쪽에서 보인 뒤 처음 경계·위험 쪽에서 보인 시각 (알림 시각 기준 몇 초 전)
                           left_safe_sec=left_safe(s['watch'].samples, stamp),
                           recent=[dict(sec=round(t - stamp, 2), conf=round(c, 2), side=side)
                                   for t, c, side in s['watch'].samples[-6:]])
                with open(self.exit_log, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + chr(10))
            except Exception as exc:
                print('[exit-log]', type(exc).__name__, exc)

        def build_state(self, cid, d, s, cfg, observed_at, video_sec, person_present, stamp):
            """R8 I3: 엔진의 사건 요약에 위치·크기·지속 시간을 덧붙인다. 판정에는 쓰지 않고 AI 입력에만 들어간다."""
            state = super().build_state(cid, d, s, cfg, observed_at, video_sec, person_present, stamp)
            try:
                state['evidence']['geometry'] = geometry_evidence(self, cid, d, s, cfg, stamp)
            except Exception as exc:   # 덧붙이기에 실패해도 원래 요약은 그대로 보낸다
                print('[ai-geometry]', type(exc).__name__, exc)
            return state

        def detector(self):
            if self.model is None:
                from ultralytics import YOLO
                self.model = DetectorProxy(YOLO(str(self.args.weights)))
            return self.model

        def detect(self, r, frame, session, cfg):
            self.detector().camera = r['cam']['cam_id']
            cam = BY_ID.get(r['cam']['cam_id'])
            if cam and cam.get('track_jump') and type(r['box_tracker']) is zj.IoUTracker:
                r['box_tracker'] = make_tracker(cam, zj)
            detections = super().detect(r, frame, session, cfg)
            return self.rejudge_people(r, frame, cfg, detections)

        def rejudge_people(self, r, frame, cfg, detections):
            """person_anchor='center'인 카메라는 사람의 구역 판정을 박스 가운데로 다시 한다.
            엔진은 발 위치로 판정한 결과를 돌려주므로, 같은 규칙(zone_relation, classify_subject)으로 덮어쓴다."""
            cam = BY_ID.get(r['cam']['cam_id'])
            if not cam or cam.get('person_anchor') != 'center' or not cfg or not cfg.get('enabled'):
                return detections
            import cv2
            from dashboard_policy import zone_relation
            h, w = frame.shape[:2]
            poly = self.pixel_polygon(cfg, w, h)
            if poly is None:
                return detections
            gate = self.gate_for(cam['cam_id'], PERSON)
            for d in detections:
                if d['name'] != PERSON:
                    continue
                fx, fy = anchor_point(cam, PERSON, d['bbox'], h)
                signed = float(cv2.pointPolygonTest(poly, (float(fx), float(fy)), True))
                relation = zone_relation(signed, gate.boundary_norm * min(w, h), False, False)
                zone, inside, danger = zj.classify_subject(
                    PERSON, signed >= 0, False, enabled=True, alert_on_exit=bool(r['cam'].get('alert_on_exit')),
                    person_inside=bool(cfg.get('person_inside')))
                if relation == 'boundary':
                    zone, danger = 'BOUNDARY', False
                d.update(anchor=(fx / max(w - 1, 1), fy / max(h - 1, 1)), inside=inside, relation=relation,
                         zone=zone, danger=danger)
            return detections

        def step_all(self):
            return render_all(self, zj.BOX_NOTE_SEC)

    return TwinAnalyzer
