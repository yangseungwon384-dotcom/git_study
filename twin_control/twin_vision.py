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
from pathlib import Path
import statistics
import time
import types

import numpy as np

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
# line_y: 관람 안전선(화면 높이 비율). 동물이 이 선에 가까워지거나 넘으면 빨강.
# near: 안전선 앞 위험 띠. person은 선을 넘어 방사장 쪽으로 들어와야 빨강.
CAMERAS = (
    dict(cam_id='ZOO_CAM_ELEPHANT_01', code='CAMERA 01', title='코끼리 방사장', desc='정상 관람 · 방사장 안 이동',
         file='elephant_cctv.mp4', species='elephant', label='코끼리', classes=('elephant',),
         min_conf=0.35, proxy=False, static_filter=True, line_y=None, near=0.0,
         alert_on_exit=False, person_inside=False, safe_zone=None),
    dict(cam_id='ZOO_CAM_GIRAFFE_01', code='CAMERA 02', title='기린 관람 구역', desc='관람객 안전선 · 구역 감시',
         file='giraffe_cctv.mp4', species='giraffe', label='기린',
         classes=('giraffe', 'elephant', 'horse', 'cow', 'sheep', 'dog', 'bird'),
         min_conf=0.05, proxy=False, static_filter=False, person_as_animal=True,
         imgsz=512, line_y=0.82, near=0.07, hold_danger=True, person_margin=0.10,
         safe_zone=np.array(
             [[70, 290], [250, 255], [620, 245], [1020, 265], [1200, 340],
              [1185, 530], [900, 555], [560, 565], [220, 540], [70, 470]], np.int32)),
    dict(cam_id='ZOO_CAM_TIGER_01', code='CAMERA 03', title='호랑이 방사장', desc='맹수 경계 · 이동 감시',
         file='tiger_cctv.mp4', species='tiger', label='호랑이',
         # YOLO에 tiger가 없어 zebra 등으로 잡은 뒤 호랑이로 묶는다. 사람은 그대로 person.
         classes=('zebra', 'horse', 'cow', 'sheep', 'cat', 'dog', 'bear', 'giraffe'),
         min_conf=0.12, proxy=True, static_filter=False, imgsz=512,
         line_y=0.70, near=0.0, person_only_line=True, person_margin=0.0,
         person_line=0.76, person_approach=True, person_inside=True, alert_on_exit=False,
         # 854x480 카카오톡 영상: 울타리 안 흙바닥이 ROI. 길(아스팔트)은 구역 밖.
         safe_zone=np.array(
             [[12, 198], [190, 182], [430, 172], [690, 178], [842, 192],
              [842, 386], [690, 396], [400, 402], [170, 398], [12, 390]], np.int32)),
    dict(cam_id='ZOO_CAM_REDPANDA_01', code='CAMERA 04', title='레서판다 관람 구역', desc='관람객·동물 접근 감시',
         file='redpanda_cctv.mp4', species='red_panda', label='레서판다',
         classes=('dog', 'cat', 'bear', 'cow', 'sheep', 'teddy bear', 'bird'),
         min_conf=0.10, proxy=True, static_filter=False, imgsz=512, line_y=0.68, near=0.0,
         person_only_line=True, person_margin=0.0, person_inside=True, person_line=0.88,
         path_guard=True, path_cy=0.73, person_center_line=0.72, person_head_line=0.50,
         stream_roi=False,
         safe_zone=np.array([[16, 10], [838, 10], [838, 398], [16, 398]], np.int32)),
)
LINE_ALERT = {
    'ZOO_CAM_GIRAFFE_01': '기린 탈출',
    'ZOO_CAM_TIGER_01': '호랑이 안전선 접근',
    'ZOO_CAM_REDPANDA_01': '레서판다 안전선 침범',
}
PERSON_ALERT = {
    'ZOO_CAM_TIGER_01': '사람 안전선 침범',
    'ZOO_CAM_REDPANDA_01': '사람 안전선 침범',
}
BY_ID = {cam['cam_id']: cam for cam in CAMERAS}
LABELS = dict({cam['species']: cam['label'] for cam in CAMERAS}, person='사람')
ZONE_KO = {'SAFE': '안전', 'DANGER': '위험', 'BOUNDARY': '경계'}


def label_of(name):
    return LABELS.get(name, name)


def is_proxy(cam_id, name):
    cam = BY_ID.get(cam_id)
    return bool(cam and cam['proxy'] and name != PERSON)


DESKTOP = Path(r'C:\Users\ysw77\OneDrive\바탕 화면')
TEAM_DIR = DESKTOP / '동물원 안전관제 팀플 영상 (1)'


def resolve_video(source_dir, name):
    """올려준 팀플 영상을 먼저 쓰고, 없으면 저장소 CCTV로 바꾼다."""
    preferred = {
        'elephant_cctv.mp4': (
            source_dir / 'team_elephant.mp4',
            TEAM_DIR / '동물원 안전관제 팀플 영상 (5).mp4',
        ),
        'tiger_cctv.mp4': (
            TEAM_DIR / 'KakaoTalk_20261013_092507818.mp4',
            source_dir / 'team_tiger.mp4',
            TEAM_DIR / '동물원 안전관제 팀플 영상 (7).mp4',
            source_dir / 'Leopard_cctv.mp4',
        ),
        'redpanda_cctv.mp4': (
            source_dir / 'team_redpanda.mp4',
            TEAM_DIR / 'toonkit-1791851341279.mp4',
            source_dir / 'panda_cctv.mp4',
        ),
        'giraffe_cctv.mp4': (
            DESKTOP / '동물원 안전관제 팀플 영상 (4).mp4',
            source_dir / 'team_giraffe.mp4',
            source_dir / 'giraffe_cctv.mp4',
        ),
    }
    for path in preferred.get(name, (source_dir / name,)):
        if path.is_file():
            return path
    return source_dir / name


def engine_cameras(original, source_dir):
    """엔진이 읽는 형식의 카메라 목록. 안전구역은 twin_vision 값을 쓴다."""
    known = {cam['cam_id']: cam for cam in original.CAMERAS}
    cameras = []
    for cam in CAMERAS:
        base = known.get(cam['cam_id'], dict(safe_zone=None, alert_on_exit=False))
        cameras.append(dict(base, code=cam['code'], title=cam['title'], desc=cam['desc'],
                            file=resolve_video(source_dir, cam['file']), cam_id=cam['cam_id'],
                            watch=(cam['species'], PERSON),
                            safe_zone=cam.get('safe_zone'),
                            alert_on_exit=bool(cam.get('alert_on_exit', True)),
                            person_inside=bool(cam.get('person_inside', True))))
    return cameras


def mark_line_danger(cam_id, detections, frame_h):
    """안전선 판정. 동물은 선에 가까워지거나 넘으면 위험, 사람은 방사장 쪽으로 넘어와야 위험."""
    cam = BY_ID.get(cam_id)
    if not cam or not detections:
        return detections
    if cam.get('line_y') is None:
        for det in detections:
            det['danger'] = False
            det['zone'] = 'SAFE'
        return detections
    line = float(cam['line_y'])
    near = float(cam.get('near', 0.06))
    person_margin = float(cam.get('person_margin', 0.10))
    person_line = float(cam.get('person_line', line))
    height = max(frame_h - 1, 1)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        foot = min(y2, frame_h - 1) / height
        if det['name'] == PERSON:
            cy = ((y1 + y2) / 2) / height
            head = y1 / height
            on_path = foot >= 0.93 or y2 >= frame_h - 8
            if cam.get('path_guard'):
                spectator = on_path and cy >= float(cam.get('path_cy', 0.73))
                crossed = (not spectator and (
                    foot < person_line - person_margin
                    or cy < float(cam.get('person_center_line', 0.72))
                    or head < float(cam.get('person_head_line', 0.50))))
                det['danger'] = crossed
                det['zone'] = 'DANGER' if crossed else 'SAFE'
                continue
            crossed = foot < person_line - person_margin and not on_path
            if cam.get('person_center') and cy < float(cam.get('person_center_line', 0.72)):
                crossed = True
            if cam.get('person_approach') and head <= line + 0.06:
                crossed = True
            if crossed:
                det['danger'] = True
                det['zone'] = 'DANGER'
        elif not cam.get('person_only_line') and foot >= line - near:
            det['danger'] = True
            det['zone'] = 'DANGER'
    return detections


def hold_escape(r, detections):
    """탈출한 그 개체만 위험을 유지한다. 뒤에 남은 기린은 안전으로 둔다."""
    cam = BY_ID.get(r['cam']['cam_id'])
    if not cam or not cam.get('hold_danger'):
        return detections
    now = time.monotonic()
    current = [d for d in detections if d.get('danger') and d['name'] != PERSON]
    if current:
        r['held_danger'] = deepcopy(current)
        r['held_until'] = now + 60.0
        return detections
    held = r.get('held_danger')
    if not held or now > r.get('held_until', 0):
        return detections
    used = set()
    for det in detections:
        if det['name'] == PERSON:
            continue
        for index, saved in enumerate(held):
            if index in used:
                continue
            if box_iou(det['bbox'], saved['bbox']) >= 0.12:
                det['danger'] = True
                det['zone'] = 'DANGER'
                used.add(index)
    for index, saved in enumerate(held):
        if index not in used:
            detections.append(deepcopy(saved))
    return detections


def update_coast(previous, detections, now):
    """직전 박스와 이어서 속도를 적어, 다음 분석 전까지 박스가 따라가게 한다."""
    coast = []
    for det in detections:
        match = max(previous, key=lambda item: box_iou(item['bbox'], det['bbox']), default=None)
        vel = [0.0, 0.0, 0.0, 0.0]
        if match and box_iou(match['bbox'], det['bbox']) >= 0.12:
            dt = max(now - match['t'], 0.04)
            vel = [(det['bbox'][i] - match['bbox'][i]) / dt for i in range(4)]
            vel = [max(-140.0, min(140.0, value)) for value in vel]
        coast.append(dict(bbox=list(det['bbox']), vel=vel, name=det['name'], danger=det['danger'],
                          confidence=det['confidence'], zone=det.get('zone') or '',
                          track_id=det.get('track_id'), t=now))
    return coast


def apply_coast(items, now):
    """분석 사이에 박스를 조금 밀어 영상에 붙인다."""
    out = []
    for item in items:
        dt = min(max(now - item['t'], 0.0), 1.2)
        bbox = [item['bbox'][i] + item['vel'][i] * dt for i in range(4)]
        out.append(dict(item, bbox=bbox))
    return out


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
            x1, y1, x2, y2 = bbox
            if (y2 - y1) < 50 or y1 < 40:
                continue
            if cam.get('person_as_animal') and y1 > 100:
                merged.append((cam['species'], confidence, bbox))
            elif confidence >= PERSON_MIN:
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
        if cam.get('imgsz'):
            options = dict(options, imgsz=cam['imgsz'])
        options.setdefault('max_det', 12)
        result = self.model.predict(frame, classes=self.real_ids[cam['cam_id']], **options)[0]
        raw = [(self.model.names[int(box.cls.item())], float(box.conf.item()),
                [float(v) for v in box.xyxy[0].tolist()])
               for box in (result.boxes if result.boxes is not None else [])]
        kept, ignored = post_process(cam, raw, self.static[cam['cam_id']], time.monotonic())
        self.ignored[cam['cam_id']] = ignored
        return [types.SimpleNamespace(boxes=[_Box(self.ids[name], confidence, bbox)
                                             for name, confidence, bbox in kept])]


_FONTS = {}
_CHIPS = {}


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


def _chip(text, color, size):
    """한글 한 줄을 작은 그림으로 만들어 둔다. 매 프레임 전체 화면을 PIL로 돌리지 않기 위함."""
    key = (text, color, size)
    chip = _CHIPS.get(key)
    if chip is None:
        from PIL import Image, ImageDraw
        font = _font(size)
        box = ImageDraw.Draw(Image.new('RGB', (8, 8))).textbbox((0, 0), text, font=font, stroke_width=2)
        image = Image.new('RGBA', (max(1, box[2] - box[0] + 6), max(1, box[3] - box[1] + 6)), (0, 0, 0, 0))
        ImageDraw.Draw(image).text((3 - box[0], 3 - box[1]), text, font=font,
                                   fill=color + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        chip = np.asarray(image)
        _CHIPS[key] = chip
    return chip


def _write(bgr, texts):
    """한글은 OpenCV 글꼴로 못 쓴다. 캐시한 글자만 얹는다. texts는 (x, y, 글, RGB 색, 크기)."""
    if not texts:
        return bgr
    h, w = bgr.shape[:2]
    for x, y, text, color, size in texts:
        chip = _chip(text, color, size)
        ch, cw = chip.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, x0 + cw), min(h, y0 + ch)
        if x1 <= x0 or y1 <= y0:
            continue
        sl = chip[:y1 - y0, :x1 - x0]
        a = sl[:, :, 3]
        if not a.any():
            continue
        rgb = sl[:, :, :3][:, :, ::-1]
        dst = bgr[y0:y1, x0:x1]
        inv = 255 - a
        blended = (rgb.astype(np.uint16) * a[..., None] + dst.astype(np.uint16) * inv[..., None]) // 255
        bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
    return bgr


def render_all(engine, box_note_sec, last_seq=None):
    """카메라별 화면용 BGR 그림. 영상은 최신 프레임을 쓰고, 박스는 마지막 분석 결과를 얹는다."""
    import cv2
    last_seq = last_seq or {}
    with engine.lock:
        shots = []
        for cid, r in engine.runtime.items():
            rendered = r['rendered']
            shots.append((cid, r['seq'], r['frame'], r['cfg'], rendered, r['session'],
                          r['banner_text'], r['banner_until'],
                          r.get('show_roi', BY_ID.get(cid, {}).get('stream_roi', True)),
                          r.get('coast') or []))
    out = []
    now = time.monotonic()
    for cid, seq, frame, cfg, rendered, session, banner_text, banner_until, show_roi, coast in shots:
        if frame is None:
            out.append((seq, None))
            continue
        if last_seq.get(cid) == seq and not (banner_text and now < banner_until):
            out.append((seq, None))
            continue
        age = now - rendered[0] if rendered else None
        sticky = bool(rendered and rendered[1] == session and rendered[2]
                      and any(d.get('danger') for d in rendered[2]))
        fresh = bool(rendered and rendered[1] == session and age is not None
                     and (age <= engine.args.max_gap or sticky))
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        texts = []
        spec = BY_ID.get(cid, {})
        crossing = bool(fresh and any(d['danger'] for d in rendered[2]))
        if spec.get('line_y') is not None:
            y = int(round(spec['line_y'] * (h - 1)))
            color = (0, 0, 255) if crossing else (0, 0, 220)
            cv2.line(canvas, (0, y), (w - 1, y), color, 4 if crossing else 2)
        if show_roi and cfg and cfg.get('polygon'):
            points = (np.asarray(cfg['polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(canvas, [points], True, (0, 220, 230), 2)
        if show_roi and cfg and cfg.get('exclude_polygon'):
            extra = (np.asarray(cfg['exclude_polygon']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(canvas, [extra], True, (0, 140, 255), 2)
        if fresh:
            drawn = apply_coast(coast, time.monotonic()) if coast else rendered[2]
            for d in drawn:
                x1, y1, x2, y2 = (int(v) for v in d['bbox'])
                color = (0, 0, 255) if d['danger'] else (0, 210, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
                zone = '위험' if d['danger'] else ZONE_KO.get(d.get('zone') or '', '안전')
                texts.append((x1, max(2, y1 - 20),
                              f"{label_of(d['name'])} {d['confidence']:.2f} {zone}",
                              (255, 80, 80) if d['danger'] else (60, 230, 60), 14))
        if banner_text and now < banner_until:
            cv2.rectangle(canvas, (0, 48), (w, 96), (0, 0, 180), 2)
            texts.append((12, 58, banner_text, (255, 255, 255), 18))
        out.append((seq, _write(canvas, texts)))
    return out


def make_analyzer(zj):
    """엔진의 LinkedAnalyzer에서 모델과 화면 그리기만 바꾼 분석기."""

    class TwinAnalyzer(zj.LinkedAnalyzer):
        def detector(self):
            if self.model is None:
                from ultralytics import YOLO
                self.model = DetectorProxy(YOLO(str(self.args.weights)))
            return self.model

        def analyze(self):
            """카메라마다 최신 프레임만 추론한다. 화면 재생은 capture가 따로 한다."""
            import os
            try:
                import torch
                if self.args.device == 'cpu':
                    torch.set_num_threads(1)
                self.detector()
            except Exception as exc:
                self.boot_error = '분석 시작 실패: ' + type(exc).__name__
                with self.lock:
                    for runtime in self.runtime.values():
                        runtime['error'] = self.boot_error
                return
            last, rotate, idle = {}, 0, 0
            ids = list(self.runtime)
            while not self.stop.is_set():
                cid = ids[rotate % len(ids)]
                rotate += 1
                r = self.runtime[cid]
                with self.lock:
                    if time.monotonic() - r['stamp'] > self.args.max_gap:
                        if r['tracks']:
                            self.reset(r, 'observation_lost')
                        continue
                    if r['frame'] is None or last.get(cid) == r['seq']:
                        idle += 1
                        if idle >= len(ids):
                            self.stop.wait(0.008)
                            idle = 0
                        continue
                    idle = 0
                    frame = r['frame'].copy()
                    stamp, session, cfg = r['stamp'], r['session'], r['cfg']
                    last[cid] = r['seq']
                    observed_at, video_sec = r['observed_at'], r['video_sec']
                try:
                    detections = self.detect(r, frame, session, cfg, stamp)
                    with self.lock:
                        if session != r['session']:
                            continue
                        finished = time.monotonic()
                        r['rendered'] = (finished, session, detections, frame)
                        r['analysis_fps'] = 1 / max(finished - r['last_analysis'], .001) if r['last_analysis'] else 0
                        r['last_analysis'] = finished
                        self.last_ok = finished
                        if finished - stamp > self.args.max_gap:
                            r['error'] = '분석 지연: 화면은 갱신하고 경고 판단은 보류'
                            continue
                        self.observe(cid, detections, frame, stamp, observed_at, video_sec, cfg)
                        r['error'] = ''
                except Exception as exc:
                    with self.lock:
                        self.reset(r, 'analysis_error')
                        r['error'] = '분석 오류: ' + type(exc).__name__

        def detect(self, r, frame, session, cfg, stamp=None):
            self.detector().camera = r['cam']['cam_id']
            detections = super().detect(r, frame, session, cfg)
            detections = mark_line_danger(r['cam']['cam_id'], detections, frame.shape[0])
            detections = hold_escape(r, detections)
            r['coast'] = update_coast(r.get('coast') or [], detections, stamp or time.monotonic())
            cid = r['cam']['cam_id']
            if any(d['danger'] and d['name'] == PERSON for d in detections) and cid in PERSON_ALERT:
                r['banner_text'] = PERSON_ALERT[cid]
                r['banner_until'] = time.monotonic() + 5.0
            elif any(d['danger'] and d['name'] != PERSON for d in detections) and cid in LINE_ALERT:
                r['banner_text'] = LINE_ALERT[cid]
                r['banner_until'] = time.monotonic() + (8.0 if BY_ID[cid].get('hold_danger') else 4.0)
            return detections

        def reset(self, runtime, reason):
            super().reset(runtime, reason)
            runtime['held_danger'] = None
            runtime['held_until'] = 0.0
            runtime['coast'] = []

        def capture(self, cid):
            """디코딩은 미리 쌓고, 화면은 고정 박자로만 내보낸다. 키프레임 끊김이 재생에 안 붙게."""
            import math
            import threading
            from collections import deque
            import cv2
            r = self.runtime[cid]
            cap = cv2.VideoCapture(str(r['cam']['file']), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(str(r['cam']['file']))
            if not cap.isOpened():
                r['error'] = '영상 파일을 열 수 없습니다.'
                cap.release()
                return
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            src_fps = src_fps if math.isfinite(src_fps) and 1 <= src_fps <= 240 else 25
            play_fps = 18.0
            ready, qlock, looped = deque(), threading.Lock(), []
            width = self.args.width
            ending = threading.Event()

            def decode():
                acc = 0.0
                while not self.stop.is_set() and not ending.is_set():
                    with qlock:
                        full = len(ready) >= 4
                    if full:
                        self.stop.wait(0.008)
                        continue
                    acc += src_fps / play_fps
                    while acc >= 2:
                        if not cap.grab():
                            break
                        acc -= 1
                    acc -= 1
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        acc = 0.0
                        with qlock:
                            ready.clear()
                            looped.append(True)
                        continue
                    h, w = frame.shape[:2]
                    if w > width:
                        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                    with qlock:
                        ready.append((frame, w, h))

            worker = threading.Thread(target=decode, daemon=True, name='decode-' + cid)
            worker.start()
            shown, last, start = 0, None, None
            try:
                while not self.stop.is_set():
                    if looped:
                        looped.clear()
                        with self.lock:
                            self.reset(r, 'video_loop')
                        shown, start, last = 0, None, None
                    if start is None:
                        start = time.monotonic()
                    due = start + shown / play_fps
                    now = time.monotonic()
                    if now < due:
                        self.stop.wait(min(due - now, 0.02))
                        continue
                    if now > due + 0.12:
                        start = now - shown / play_fps
                    with qlock:
                        item = ready.popleft() if ready else None
                    if item is None:
                        if last is None:
                            self.stop.wait(0.01)
                            continue
                        frame, native_w, native_h = last
                    else:
                        last = item
                        frame, native_w, native_h = item
                    with self.lock:
                        r['native_size'] = (native_w, native_h)
                        if r['cfg'] is None:
                            old = r['cam']['safe_zone']
                            polygon = [[float(x) / (native_w - 1), float(y) / (native_h - 1)]
                                       for x, y in old] if old is not None else []
                            valid = bool(polygon) and all(0 <= v <= 1 for p in polygon for v in p)
                            r['cfg'] = dict(polygon=polygon if valid else [],
                                            enabled=valid and bool(r['cam'].get('alert_on_exit')
                                                                  or r['cam'].get('person_inside')),
                                            person_inside=bool(r['cam'].get('person_inside')),
                                            reviewed=False, version=0)
                        r.update(frame=frame, stamp=time.monotonic(), observed_at=zj.now().isoformat(),
                                 video_sec=shown / play_fps, seq=r['seq'] + 1)
                    shown += 1
            except Exception as exc:
                r['error'] = '영상 오류: ' + type(exc).__name__
            finally:
                ending.set()
                worker.join(timeout=1.0)
                cap.release()

        def step_all(self, last_seq=None):
            return render_all(self, zj.BOX_NOTE_SEC, last_seq)

    return TwinAnalyzer
