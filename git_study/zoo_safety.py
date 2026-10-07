# 실습 — 기린 CCTV 탐지 + 철창 ROI + 이벤트 로그 · 관제 알림
# 목표:
#   1) giraffe_cctv.mp4 객체 탐지
#   2) 철창 안=SAFE / 철창 밖=DANGER
#   3) SAFE→DANGER 전환 시 이벤트 로그 저장 + 관제센터 알림
#
# 알림 문구: "위험구역을 벗어났으니 다른 CCTV를 확인하세요"
# 조작: q 종료

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

# Windows + Anaconda: OpenMP 중복 로딩 우회
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 경로 · 상수
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
VIDEO = BASE / "giraffe_cctv.mp4"
LOG_DIR = BASE / "logs"
SNAP_DIR = BASE / "snapshots"
EVENT_CSV = LOG_DIR / "zoo_events.csv"
ALERT_INBOX = LOG_DIR / "control_center_inbox.txt"

CAMERA_ID = "ZOO_CAM_GIRAFFE_01"
ALERT_MSG = "위험구역을 벗어났으니 다른 CCTV를 확인하세요"
WATCH_NAMES = ("giraffe", "person")

CONF = 0.1
LOST_MAX = 20
BOTTOM_MARGIN = 8
ALERT_HOLD_SEC = 4.0
FALLBACK_ID_START = 9000
WINDOW_TITLE = "zoo ROI + event log + control alert"

SAFE_ZONE = np.array(
    [
        [40, 300],
        [180, 230],
        [420, 200],
        [700, 210],
        [830, 280],
        [820, 360],
        [650, 420],
        [430, 440],
        [180, 400],
        [50, 350],
    ],
    dtype=np.int32,
)

COLOR_SAFE = (0, 200, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_BOX_SAFE = (0, 255, 0)
COLOR_BOX_DANGER = (0, 0, 255)
COLOR_BANNER = (0, 0, 180)
COLOR_HUD = (240, 240, 240)

EVENT_FIELDS = [
    "detected_at",
    "camera_id",
    "track_id",
    "object",
    "zone",
    "event_type",
    "severity",
    "message",
    "snapshot_path",
    "confidence",
]

ZoneName = str  # "SAFE" | "DANGER"


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
@dataclass
class TrackInfo:
    xyxy: list
    name: str
    conf: float
    lost: int = 0


@dataclass
class TrackerState:
    memory: Dict[int, TrackInfo] = field(default_factory=dict)
    prev_zone: Dict[int, ZoneName] = field(default_factory=dict)
    alerted: Set[int] = field(default_factory=set)
    next_fallback_id: int = FALLBACK_ID_START
    alert_count: int = 0

    def reset_loop(self) -> None:
        """영상 한 바퀴 끝났을 때 트랙·알림 상태 초기화"""
        self.memory.clear()
        self.prev_zone.clear()
        self.alerted.clear()

    def prune_lost(self) -> None:
        dead = [tid for tid, info in self.memory.items() if info.lost > LOST_MAX]
        for tid in dead:
            self.memory.pop(tid, None)
            self.prev_zone.pop(tid, None)


@dataclass
class BannerState:
    until: float = 0.0
    text: str = ""

    def show(self, video_time: float, text: str, hold: float = ALERT_HOLD_SEC) -> None:
        self.until = video_time + hold
        self.text = text

    def active(self, video_time: float) -> bool:
        return bool(self.text) and video_time <= self.until


# ---------------------------------------------------------------------------
# 준비 · I/O
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_CSV.exists():
        with open(EVENT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=EVENT_FIELDS).writeheader()


def load_model() -> Tuple[YOLO, list]:
    weights = BASE.parent / "unit5" / "yolo26n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolo26n.pt")
    watch = [i for i, name in model.names.items() if name in WATCH_NAMES]
    return model, watch


def open_video(path: Path) -> Tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    delay = max(1, int(1000 / fps))
    return cap, delay


def save_snapshot(frame, tid: int, when: datetime) -> str:
    name = f"alert_{when.strftime('%Y%m%d_%H%M%S')}_id{tid}.jpg"
    path = SNAP_DIR / name
    cv2.imwrite(str(path), frame)
    return str(path.relative_to(BASE)).replace("\\", "/")


def append_event_log(row: dict) -> None:
    with open(EVENT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=EVENT_FIELDS).writerow(row)


def send_to_control_center(message: str, row: dict) -> str:
    """관제센터 알림 (실습: 파일 + 콘솔). 실환경이면 HTTP/웹훅으로 교체."""
    line = (
        f"[{row['detected_at']}] [{row['camera_id']}] "
        f"track={row['track_id']} {row['object']} "
        f"→ {message}\n"
        f"  event={row['event_type']} severity={row['severity']} "
        f"snapshot={row['snapshot_path']}\n"
    )
    with open(ALERT_INBOX, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("-" * 60 + "\n")
    print("\n[관제센터 알림]")
    print(line.strip())
    return line


def build_danger_event(tid: int, name: str, conf: float, snap: str, when: datetime) -> dict:
    return {
        "detected_at": when.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": CAMERA_ID,
        "track_id": tid,
        "object": name,
        "zone": "철창밖_위험구역",
        "event_type": "ZONE_EXIT_DANGER",
        "severity": "HIGH",
        "message": ALERT_MSG,
        "snapshot_path": snap,
        "confidence": round(conf, 3),
    }


# ---------------------------------------------------------------------------
# 구역 판정
# ---------------------------------------------------------------------------
def anchor_xy(xyxy, frame_h: int) -> Tuple[float, float]:
    x1, _y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, min(float(y2), float(frame_h - 1))


def in_poly(px: float, py: float, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(px), float(py)), False) >= 0


def zone_label(px: float, py: float, frame_h: int) -> Tuple[ZoneName, tuple]:
    if py >= frame_h - BOTTOM_MARGIN:
        return "DANGER", COLOR_DANGER
    if in_poly(px, py, SAFE_ZONE):
        return "SAFE", COLOR_SAFE
    return "DANGER", COLOR_DANGER


def is_zone_exit(before: ZoneName, after: ZoneName) -> bool:
    """SAFE(또는 미기록) → DANGER 전이"""
    return before != "DANGER" and after == "DANGER"


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------
def draw_rois(frame) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    danger_overlay = out.copy()
    cv2.rectangle(danger_overlay, (0, 0), (w - 1, h - 1), COLOR_DANGER, -1)
    out = cv2.addWeighted(danger_overlay, 0.12, out, 0.88, 0)

    safe_overlay = out.copy()
    cv2.fillPoly(safe_overlay, [SAFE_ZONE], COLOR_SAFE)
    out = cv2.addWeighted(safe_overlay, 0.25, out, 0.75, 0)
    cv2.polylines(out, [SAFE_ZONE], True, COLOR_SAFE, 2)

    cv2.putText(out, "SAFE (inside)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SAFE, 2)
    cv2.putText(out, "DANGER (outside)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DANGER, 2)
    return out


def draw_alert_banner(canvas, text: str) -> None:
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 70), (w, 130), COLOR_BANNER, -1)
    canvas[:] = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
    cv2.putText(canvas, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_track(canvas, tid: int, info: TrackInfo, zname: ZoneName, zcolor: tuple) -> None:
    x1, y1, x2, y2 = info.xyxy
    held = info.lost > 0
    box_color = COLOR_BOX_SAFE if zname == "SAFE" else COLOR_BOX_DANGER
    thickness = 1 if held else 2

    fx, fy = anchor_xy(info.xyxy, canvas.shape[0])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
    cv2.circle(canvas, (int(fx), int(fy)), 5, zcolor, -1)

    tag = f"ID{tid} {info.name} {info.conf:.2f} | {zname}"
    if held:
        tag += " (hold)"
    cv2.putText(canvas, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


def draw_hud(canvas, n_safe: int, n_danger: int, alert_count: int) -> None:
    h = canvas.shape[0]
    text = f"SAFE:{n_safe} DANGER:{n_danger} alerts:{alert_count}  q=quit"
    cv2.putText(canvas, text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD, 2)


# ---------------------------------------------------------------------------
# 트래킹 · 이벤트
# ---------------------------------------------------------------------------
def update_tracks(result, model: YOLO, state: TrackerState) -> None:
    seen: Set[int] = set()
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        has_id = boxes.id is not None
        for i, box in enumerate(boxes):
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            tid = int(boxes.id[i]) if has_id else state.next_fallback_id + i
            state.memory[tid] = TrackInfo(xyxy=xyxy, name=name, conf=conf, lost=0)
            seen.add(tid)

    for tid, info in state.memory.items():
        if tid not in seen:
            info.lost += 1

    state.prune_lost()


def handle_zone_exit(
    frame,
    tid: int,
    info: TrackInfo,
    video_time: float,
    state: TrackerState,
    banner: BannerState,
) -> None:
    now = datetime.now()
    snap = save_snapshot(frame, tid, now)
    row = build_danger_event(tid, info.name, info.conf, snap, now)
    append_event_log(row)
    send_to_control_center(ALERT_MSG, row)
    state.alerted.add(tid)
    state.alert_count += 1
    banner.show(video_time, ALERT_MSG)


def process_frame(
    frame,
    model: YOLO,
    watch: list,
    state: TrackerState,
    banner: BannerState,
    video_time: float,
) -> np.ndarray:
    result = model.track(
        frame,
        conf=CONF,
        classes=watch or None,
        persist=True,
        verbose=False,
    )[0]
    update_tracks(result, model, state)

    canvas = draw_rois(frame)
    h = frame.shape[0]
    n_safe = n_danger = 0

    for tid, info in state.memory.items():
        fx, fy = anchor_xy(info.xyxy, h)
        zname, zcolor = zone_label(fx, fy, h)

        before = state.prev_zone.get(tid, "SAFE")
        if is_zone_exit(before, zname) and tid not in state.alerted:
            handle_zone_exit(frame, tid, info, video_time, state, banner)

        state.prev_zone[tid] = zname
        if zname == "SAFE":
            n_safe += 1
        else:
            n_danger += 1

        draw_track(canvas, tid, info, zname, zcolor)

    if banner.active(video_time):
        draw_alert_banner(canvas, banner.text)

    draw_hud(canvas, n_safe, n_danger, state.alert_count)
    return canvas


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------
def run() -> None:
    ensure_dirs()
    model, watch = load_model()

    print("감시 클래스:", [model.names[i] for i in watch])
    print("영상:", VIDEO.name)
    print("이벤트 로그:", EVENT_CSV)
    print("관제 알림함:", ALERT_INBOX)
    print("조작: q 종료\n")

    cap, delay = open_video(VIDEO)
    state = TrackerState()
    banner = BannerState()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.reset_loop()
                continue

            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            canvas = process_frame(frame, model, watch, state, banner, video_time)

            cv2.imshow(WINDOW_TITLE, canvas)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n종료")
    print("이벤트 로그:", EVENT_CSV)
    print("관제 알림함:", ALERT_INBOX)
    print("알림 횟수:", state.alert_count)


if __name__ == "__main__":
    run()
