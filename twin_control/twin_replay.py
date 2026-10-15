"""영상 재생 시험: 서버와 같은 탐지·구역 판정 규칙으로 영상을 다시 돌려, 이탈 알림이 언제 왜 나는지 기록한다.

    python twin_replay.py --camera ZOO_CAM_ELEPHANT_01                 저장된 구역으로 지금 규칙과 이전 규칙 비교
    python twin_replay.py --camera ZOO_CAM_ELEPHANT_01 --shrink 0.6     구역을 60%로 줄여 실제 이탈 장면을 만든 뒤 비교
    python twin_replay.py --camera ZOO_CAM_ELEPHANT_01 --events out.jsonl   알림마다 근거를 파일로 남긴다

서버가 쓰는 엔진 부품(IoUTracker, zone_relation, ObjectWatch)과 twin_vision의 후처리를 그대로 쓴다.
영상은 서버처럼 초당 약 3장만 분석하고, 영상이 처음으로 돌아가면 추적을 다시 시작한다.
YOLO 결과는 replay_cache/에 저장해 두고 다시 쓴다 (규칙만 바꿔 여러 번 비교할 수 있게).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'zoo_safe'))

import twin_vision  # noqa: E402

SOURCE = Path(r'C:\code\code_test\zoo')
ANALYSIS_FPS = 3.0
CACHE = HERE / 'replay_cache'
# 'v1'(이전 규칙)으로 돌릴 때 빼는 카메라별 설정. 엔진 기본 규칙과 같아진다.
RULE_KEYS = ('person_min', 'person_gate', 'person_anchor', 'exit_samples', 'exit_sec', 'boundary_norm')


def detections(cam, weights, width=960):
    """프레임별 YOLO 원시 결과 (카메라의 클래스와 사람만). 캐시가 있으면 그것을 쓴다."""
    import cv2
    CACHE.mkdir(exist_ok=True)
    video = SOURCE / cam['file']
    key = CACHE / f"{cam['cam_id']}_{video.stat().st_size}_{int(ANALYSIS_FPS * 10)}.json"
    if key.is_file():
        return json.loads(key.read_text(encoding='utf-8'))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    real = {name: index for index, name in model.names.items()}
    ids = [real[n] for n in (*cam['classes'], 'person') if n in real]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / ANALYSIS_FPS))
    frames = []
    for index in range(0, count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, round(h * width / w)))
            h, w = frame.shape[:2]
        result = model.predict(frame, classes=ids, conf=0.1, imgsz=640, device='cpu', verbose=False)[0]
        frames.append(dict(t=index / fps, size=[w, h], raw=[
            [model.names[int(b.cls.item())], round(float(b.conf.item()), 4), [round(float(v), 1) for v in b.xyxy[0].tolist()]]
            for b in (result.boxes if result.boxes is not None else [])]))
    cap.release()
    data = dict(fps=fps, duration=count / fps, frames=frames)
    key.write_text(json.dumps(data), encoding='utf-8')
    return data


def zone(cam_id, data_dir, shrink):
    with sqlite3.connect(data_dir / 'events.sqlite3') as db:
        row = db.execute('SELECT json FROM roi WHERE camera=?', (cam_id,)).fetchone()
    if not row:
        raise SystemExit('이 카메라에는 저장된 구역이 없습니다. 카메라 탭에서 구역을 먼저 그려 주세요.')
    cfg = json.loads(row[0])
    if shrink != 1.0:
        xs = [p[0] for p in cfg['polygon']]; ys = [p[1] for p in cfg['polygon']]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        cfg['polygon'] = [[cx + (x - cx) * shrink, cy + (y - cy) * shrink] for x, y in cfg['polygon']]
    return cfg


def replay(cam, data, cfg, rules, loops, start=0.0):
    """rules: 'v1' (이전 규칙) 또는 'v2' (twin_vision의 카메라별 규칙). 알림 목록을 돌려준다."""
    import cv2
    import numpy as np
    import zoo_vision_jev_new as zj
    from dashboard_policy import DemoGate, ObjectWatch, zone_relation
    base_gate = DemoGate(candidate_min=0.1, min_span_sec=1.0, max_gap_sec=3.0)
    gate = twin_vision.camera_gate(cam['cam_id'], 'elephant', base_gate) if rules == 'v2' else base_gate
    watch_class = twin_vision.object_watch_class() if rules == 'v2' else ObjectWatch
    margin_px = 8          # zoo_safety.BOTTOM_MARGIN
    rule_cam = cam if rules == 'v2' else {k: v for k, v in cam.items() if k not in RULE_KEYS}
    events, static = [], twin_vision.StaticFilter()
    duration = data['duration']
    for loop in range(loops):
        tracker, watches, last = zj.IoUTracker(), {}, {}
        for frame_index, frame in enumerate(data['frames']):
            if frame['t'] < start:
                continue
            t = loop * duration + frame['t']
            w, h = frame['size']
            poly = (np.asarray(cfg['polygon'], np.float32) * [w - 1, h - 1]).astype(np.int32)
            raw = [tuple(r) for r in frame['raw']]
            kept, _ = twin_vision.post_process(rule_cam, raw, static, t)
            dets = tracker.update([dict(name=n, confidence=c, bbox=[int(v) for v in b]) for n, c, b in kept])
            for d in dets:
                name, tid = d['name'], d['track_id']
                if name == 'person':
                    g = twin_vision.camera_gate(cam['cam_id'], name, base_gate) if rules == 'v2' else base_gate
                else:
                    g = gate
                fx, fy = twin_vision.anchor_point(rule_cam, name, d['bbox'], h)
                signed = float(cv2.pointPolygonTest(poly, (fx, fy), True))
                relation = zone_relation(signed, g.boundary_norm * min(w, h), fy >= h - margin_px, False)
                s = watches.get(tid)
                if s and (t - last[tid] > g.max_gap_sec or s['name'] != name):
                    s = None
                if s is None:
                    s = watches[tid] = dict(name=name, watch=watch_class(), samples=[], crossed=None)
                last[tid] = t
                if rules == 'v2':
                    watch_class.camera = cam['cam_id']
                out = s['watch'].update(t, (loop, frame_index), d['confidence'], relation, g, bool(cfg.get('person_inside')), name)
                if s['crossed'] is None and ((name == 'person') == (signed > 0)):
                    s['crossed'] = t     # 발 위치가 처음으로 선을 넘은 관측 (사람은 안으로, 동물은 밖으로)
                s['samples'].append(dict(t=round(t, 2), conf=round(d['confidence'], 2), relation=relation, dist=round(signed, 1)))
                del s['samples'][:-8]
                if out['action'] == 'open':
                    events.append(dict(loop=loop, t=round(t, 2), video_sec=round(frame['t'], 2), track=tid, object=name,
                                       kind=out['kind'], confidence=round(d['confidence'], 3), relation=relation,
                                       distance_px=round(signed, 1), margin_px=round(g.boundary_norm * min(w, h), 1),
                                       bbox=d['bbox'], foot=[round(fx), round(fy)], recent=list(s['samples']),
                                       delay=round(t - s['crossed'], 2) if s['crossed'] is not None else None))
    return events


def summary(events, label, loops, duration):
    animal = [e for e in events if e['object'] != 'person']
    person = [e for e in events if e['object'] == 'person']
    print(f'[{label}] {loops}회 반복 ({loops * duration:.0f}초): 동물 이탈 알림 {len(animal)}건, 사람 진입 알림 {len(person)}건')
    return animal + person


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', default='ZOO_CAM_ELEPHANT_01')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'twin_data')
    parser.add_argument('--weights', type=Path, default=Path(r'C:\code\code_test\unit5\yolo26n.pt'))
    parser.add_argument('--loops', type=int, default=3)
    parser.add_argument('--shrink', type=float, default=1.0, help='구역 크기 배율. 1보다 작으면 실제 이탈 장면을 만든다')
    parser.add_argument('--events', type=Path, help='알림별 근거를 저장할 jsonl')
    parser.add_argument('--start', type=float, default=0.0, help='영상의 이 초부터 분석한다 (사람이 이미 구역 안에서 처음 잡히는 상황 만들기)')
    parser.add_argument('--person-anchor', choices=('foot', 'center'), help='v2 규칙에서 사람 판정 기준점을 바꿔 시험')
    parser.add_argument('--margin', type=float, help='v2 규칙의 경계 완충 폭을 이 값(화면 짧은 변 대비 비율)으로 바꿔 시험')
    args = parser.parse_args()
    cam = twin_vision.BY_ID[args.camera]
    if args.person_anchor:
        cam['person_anchor'] = args.person_anchor
    if args.margin is not None:
        cam['boundary_norm'] = args.margin
    data = detections(cam, args.weights)
    cfg = zone(args.camera, args.data_dir, args.shrink)
    print(f"[시작 {args.start}초] {cam['title']}: 분석 프레임 {len(data['frames'])}장 / 영상 {data['duration']:.1f}초, 구역 배율 {args.shrink}")
    results = {}
    for rules in ('v1', 'v2'):
        results[rules] = replay(cam, data, cfg, rules, args.loops, args.start)
        animal = summary(results[rules], rules, args.loops, data['duration'])
        for e in animal:
            tail = ' '.join(f"{x['relation'][:3]}{x['conf']:.2f}" for x in e['recent'][-5:])
            print(f"   {e['object']:<8} {e['video_sec']:5.1f}s #{e['track']:<3} {e['kind']:<10} 넘은 뒤 {e['delay']}초 신뢰도 {e['confidence']:.2f} 경계까지 {e['distance_px']:6.1f}px (완충 {e['margin_px']}px) 최근: {tail}")
    if args.events:
        with open(args.events, 'w', encoding='utf-8') as handle:
            for rules, events in results.items():
                for e in events:
                    handle.write(json.dumps(dict(e, rules=rules), ensure_ascii=False) + '\n')
        print('알림별 근거 저장:', args.events)


if __name__ == '__main__':
    main()
