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

# species: 엔진과 사건 기록에 남는 이름.  label: 화면에 쓰는 이름.
# classes: 그 종으로 묶을 YOLO 클래스.  proxy: 묶은 클래스가 실제 종명이 아니다.
# static_filter: 정지 물체 제외를 쓴다. 가만히 있는 동물이 낮은 신뢰도로 잡히는 카메라에서는 끈다.
# enclosure: 동물이 있어야 할 쪽. inside는 그려 둔 구역 안이 우리.
# outside는 그려 둔 구역이 관람 쪽이라, 우리 안은 그 다각형 밖이다.
# near_fence: 우리 안 동물이 울타리에 붙고 관람객도 그 아래에 있으면 경계.
# person_as_animal: 멀리 있는 동물이 person으로 잡히면 그 카메라의 종으로 되돌린다.
# dirt_boundary: 그려 둔 ROI(흙바닥) 안에 온전한 사람 박스의 발이 들어온 경우만 경계로 본다.
CAMERAS = (
    dict(cam_id='ZOO_CAM_ELEPHANT_01', code='CAMERA 01', title='코끼리 방사장', desc='대형 동물 경계 · 이동 감시',
         file='elephant_cctv.mp4', species='elephant', label='코끼리', classes=('elephant',),
         min_conf=0.45, proxy=False, static_filter=True, enclosure='inside', near_fence=True),
    dict(cam_id='ZOO_CAM_GIRAFFE_01', code='CAMERA 02', title='기린 관람 구역', desc='관람객 안전선 · 구역 감시',
         file='giraffe_cctv.mp4', species='giraffe', label='기린', classes=('giraffe',),
         min_conf=0.15, person_min=0.20, proxy=False, static_filter=False,
         enclosure='inside', person_as_animal=True),
    dict(cam_id='ZOO_CAM_TIGER_01', code='CAMERA 03', title='호랑이 방사장', desc='맹수 경계 · 이동 감시',
         file='tiger_cctv.mp4', species='tiger', label='호랑이 추정',
         classes=('zebra', 'giraffe', 'cow', 'sheep', 'horse', 'cat', 'dog', 'bear'),
         min_conf=0.35, proxy=True, static_filter=False, enclosure='outside', dirt_boundary=True),
    dict(cam_id='ZOO_CAM_REDPANDA_01', code='CAMERA 04', title='레서판다 관람 구역', desc='관람객·동물 접근 감시',
         file='redpanda_cctv.mp4', species='red_panda', label='레서판다 추정',
         classes=('dog', 'cat', 'bear', 'cow', 'sheep', 'teddy bear'),
         min_conf=0.25, proxy=True, static_filter=False, enclosure='outside',
         climber_height=0.18),
)
BY_ID = {cam['cam_id']: cam for cam in CAMERAS}
LABELS = dict({cam['species']: cam['label'] for cam in CAMERAS}, person='사람')
ZONE_KO = {'SAFE': '안전', 'DANGER': '위험', 'BOUNDARY': '경계'}


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
    for name, confidence, bbox in raw:
        if name == PERSON:
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
    """카메라별 화면용 RGB 그림. 바탕은 방금 읽은 영상 프레임이고, 박스는 마지막 분석 결과다."""
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
        fresh = bool(rendered and rendered[1] == session and rendered[2] is not None and age <= engine.args.max_gap)
        # 분석이 끝난 옛 화면을 붙잡지 않는다. 재생 속도는 파일에서 읽은 최신 프레임을 따른다.
        canvas = (frame if frame is not None else shot).copy()
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
                zone = d.get('zone') or ''
                if zone == 'BOUNDARY':
                    color, text_color = (0, 165, 255), (255, 190, 40)
                elif d['danger']:
                    color, text_color = (0, 0, 255), (255, 80, 80)
                else:
                    color, text_color = (0, 210, 0), (60, 230, 60)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                texts.append((x1, max(2, y1 - 20), f"{label_of(d['name'])} {d['confidence']:.2f}"
                              + (' ' + ZONE_KO.get(zone, '') if zone else ''),
                              text_color, 14))
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


def _polygon(cfg, w, h):
    import numpy as np
    if not cfg or len(cfg.get('polygon') or []) < 3:
        return None
    return (np.asarray(cfg['polygon'], np.float32) * [w - 1, h - 1]).astype(np.int32)


def _center(det):
    x1, y1, x2, y2 = det['bbox']
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _foot(det, h):
    x1, _y1, x2, y2 = det['bbox']
    return (x1 + x2) / 2.0, min(float(y2), h - 1)


def _mark(det, zone, danger, relation, inside):
    det['zone'], det['danger'], det['relation'], det['inside'] = zone, danger, relation, inside


def _box_height(det):
    return det['bbox'][3] - det['bbox'][1]


def _mostly_inside(inner, outer):
    """작은 박스가 큰 박스 안에 얼마나 들어가 있는지. 겹쳐 생긴 조각이면 1에 가깝다."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _person_fragment(det, people):
    """겹치면서 작게 잡힌 사람 박스. 그 아랫변은 발이 아니다."""
    height = _box_height(det)
    tallest = max(_box_height(other) for other in people)
    if tallest and height < tallest * 0.62:
        return True
    for other in people:
        if other is det or _box_height(other) <= height:
            continue
        if _mostly_inside(det['bbox'], other['bbox']) >= 0.5:
            return True
    return False


def retune(cam_id, frame, cfg, detections):
    """카메라마다 우리 안/밖과 오탐을 다시 매긴다. 엔진의 기본은 발이 다각형 안이면 안전이다."""
    import cv2
    cam = BY_ID.get(cam_id)
    if not cam or not detections:
        return detections
    h, w = frame.shape[:2]
    poly = _polygon(cfg, w, h)
    if poly is None:
        return detections
    short = min(w, h)

    def dist(pt):
        return float(cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True))

    if cam.get('person_as_animal'):
        for det in detections:
            if det['name'] == PERSON and dist(_center(det)) >= 0:
                det['name'] = cam['species']
                _mark(det, 'SAFE', False, 'inside', True)
    if cam.get('near_fence'):
        near_animal, near_person = 0.12 * short, 0.22 * short
        elephants, people = [], []
        for det in detections:
            foot_dist = dist(_foot(det, h))
            if det['name'] == cam['species'] and foot_dist >= 0 and foot_dist <= near_animal:
                elephants.append(det)
            elif (det['name'] == PERSON and -near_person <= foot_dist < 0
                  and _foot(det, h)[1] >= h * 0.72):
                people.append(det)
        if elephants and people:
            for det in elephants + people:
                _mark(det, 'BOUNDARY', False, 'boundary', det['name'] != PERSON)
    if cam.get('enclosure') == 'outside':
        climber_h = cam['climber_height'] * h if 'climber_height' in cam else None
        people = [det for det in detections if det['name'] == PERSON]
        crossed = False
        for det in detections:
            habitat = dist(_center(det)) < 0
            foot_in_roi = dist(_foot(det, h)) >= 0
            height = det['bbox'][3] - det['bbox'][1]
            partial = cam.get('dirt_boundary') and people and _person_fragment(det, people)
            if det['name'] == PERSON and cam.get('dirt_boundary') and foot_in_roi and not partial:
                _mark(det, 'BOUNDARY', False, 'boundary', True)
            elif det['name'] == PERSON and cam.get('dirt_boundary'):
                _mark(det, 'SAFE', False, 'outside', False)
            elif det['name'] == PERSON and climber_h is not None and habitat and height < climber_h:
                det['name'] = cam['species']
                _mark(det, 'SAFE', False, 'inside', True)
            elif det['name'] == PERSON and climber_h is not None and habitat:
                _mark(det, 'DANGER', True, 'inside', True)
                crossed = True
            elif det['name'] == PERSON:
                _mark(det, 'SAFE', False, 'outside', False)
            elif habitat:
                _mark(det, 'SAFE', False, 'inside', True)
            else:
                _mark(det, 'DANGER', True, 'outside', False)
        if crossed:
            cfg['person_inside'] = True
    return detections


def make_analyzer(zj):
    """엔진의 LinkedAnalyzer에서 모델과 화면 그리기만 바꾼 분석기."""

    class TwinAnalyzer(zj.LinkedAnalyzer):
        def detector(self):
            if self.model is None:
                from ultralytics import YOLO
                self.model = DetectorProxy(YOLO(str(self.args.weights)))
            return self.model

        def detect(self, r, frame, session, cfg):
            self.detector().camera = r['cam']['cam_id']
            return retune(r['cam']['cam_id'], frame, cfg, super().detect(r, frame, session, cfg))

        def step_all(self):
            return render_all(self, zj.BOX_NOTE_SEC)

    return TwinAnalyzer
