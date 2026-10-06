# ═══════════════════════════════════════════════
# Zoo Vision — SafeZoo 안전관제 통합 관제보드
# 강의 매핑: 04_06 통합 관제 대시보드 (실습 123~127)
# 디자인: safezoo.css
# 데이터: zoo_safety.py (탐지·ROI) + logs/
#
# 실행: python zoo_vision.py
# ═══════════════════════════════════════════════

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import gradio as gr
import numpy as np
import pandas as pd

import zoo_safety as zs

# ---------------------------------------------------------------------------
# 경로 · 상수
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
SNAP_DIR = BASE / "snapshots"
EVENT_CSV = LOG_DIR / "zoo_events.csv"
ALERT_INBOX = LOG_DIR / "control_center_inbox.txt"
CSS_PATH = BASE / "safezoo.css"

CAMERA_ID = "ZOO_CAM_MULTI"
REFRESH_SEC = 5
DETECT_SEC = 0.4
SNAPSHOT_LIMIT = 6
TABLE_LIMIT = 50

GIRAFFE_ZONE = zs.SAFE_ZONE.copy()


def full_frame_safe_zone(w: int, h: int) -> np.ndarray:
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.int32,
    )


CAMERAS = [
    {
        "code": "CAMERA 01",
        "title": "코끼리 방사장",
        "desc": "대형 동물 경계 · 이동 감시",
        "file": BASE / "elephant_cctv.mp4",
        "cam_id": "ZOO_CAM_ELEPHANT_01",
        "watch": ("elephant", "person"),
        "safe_zone": None,
        "alert_on_exit": False,
    },
    {
        "code": "CAMERA 02",
        "title": "기린 관람 구역",
        "desc": "관람객 안전선 · 구역 감시",
        "file": BASE / "giraffe_cctv.mp4",
        "cam_id": "ZOO_CAM_GIRAFFE_01",
        "watch": ("giraffe", "person"),
        "safe_zone": GIRAFFE_ZONE,
        "alert_on_exit": True,
    },
    {
        "code": "CAMERA 03",
        "title": "표범 방사장",
        "desc": "맹수 경계 · 이동 감시",
        "file": BASE / "Leopard_cctv.mp4",
        "cam_id": "ZOO_CAM_LEOPARD_01",
        "watch": ("cat", "dog", "bear", "person"),
        "safe_zone": None,
        "alert_on_exit": False,
    },
    {
        "code": "CAMERA 04",
        "title": "판다 관람 구역",
        "desc": "관람객·동물 접근 감시",
        "file": BASE / "panda_cctv.mp4",
        "cam_id": "ZOO_CAM_PANDA_01",
        "watch": ("bear", "person"),
        "safe_zone": None,
        "alert_on_exit": False,
    },
]

EVENT_COLS = [
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

DISPLAY_COLS = {
    "detected_at": "시각",
    "camera_id": "카메라",
    "track_id": "Track",
    "object": "객체",
    "zone": "구역",
    "event_type": "유형",
    "severity": "심각도",
    "message": "메시지",
    "confidence": "신뢰도",
}


# ---------------------------------------------------------------------------
# 멀티캠 분석기
# ---------------------------------------------------------------------------
class MultiCamAnalyzer:
    def __init__(self) -> None:
        self.model = None
        self.caps: Dict[str, cv2.VideoCapture] = {}
        self.fps: Dict[str, float] = {}
        self.states: Dict[str, zs.TrackerState] = {}
        self.banners: Dict[str, zs.BannerState] = {}
        self.ready = False

    def _ensure(self) -> None:
        if self.ready:
            return
        zs.ensure_dirs()
        self.model, _ = zs.load_model()
        for cam in CAMERAS:
            cid = cam["cam_id"]
            self.states[cid] = zs.TrackerState()
            self.banners[cid] = zs.BannerState()
            path = cam["file"]
            if path.exists():
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    self.caps[cid] = cap
                    self.fps[cid] = cap.get(cv2.CAP_PROP_FPS) or 24
                    if cam["safe_zone"] is None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
                        cam["safe_zone"] = full_frame_safe_zone(w, h)
        self.ready = True

    def _watch_ids(self, names: Tuple[str, ...]) -> list:
        return [i for i, n in self.model.names.items() if n in names]

    def _read_frame(self, cid: str):
        cap = self.caps.get(cid)
        if cap is None:
            return None
        skip = max(0, int(self.fps.get(cid, 24) * DETECT_SEC) - 1)
        for _ in range(skip):
            if not cap.grab():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.states[cid].reset_loop()
                break
        ok, frame = cap.retrieve() if skip > 0 else cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.states[cid].reset_loop()
            ok, frame = cap.read()
            if not ok:
                return None
        return frame

    def step_all(self) -> List[Optional[np.ndarray]]:
        self._ensure()
        out: List[Optional[np.ndarray]] = []
        for cam in CAMERAS:
            cid = cam["cam_id"]
            if not cam["file"].exists() or cid not in self.caps:
                out.append(None)
                continue

            frame = self._read_frame(cid)
            if frame is None:
                out.append(None)
                continue

            zone = cam.get("safe_zone")
            if zone is None:
                h, w = frame.shape[:2]
                zone = full_frame_safe_zone(w, h)
                cam["safe_zone"] = zone

            prev_zone = zs.SAFE_ZONE
            prev_cam = zs.CAMERA_ID
            prev_bottom = zs.BOTTOM_MARGIN
            zs.SAFE_ZONE = np.asarray(zone, dtype=np.int32)
            zs.CAMERA_ID = cid
            zs.BOTTOM_MARGIN = zs.BOTTOM_MARGIN if cam.get("alert_on_exit") else 0

            watch = self._watch_ids(tuple(cam["watch"]))
            t_ms = self.caps[cid].get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if cam.get("alert_on_exit"):
                canvas = zs.process_frame(
                    frame, self.model, watch, self.states[cid], self.banners[cid], t_ms
                )
            else:
                canvas = self._preview_frame(
                    frame, watch, self.states[cid], self.banners[cid], t_ms
                )

            zs.SAFE_ZONE = prev_zone
            zs.CAMERA_ID = prev_cam
            zs.BOTTOM_MARGIN = prev_bottom
            out.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return out

    def _preview_frame(self, frame, watch, state, banner, video_time):
        result = self.model.track(
            frame,
            conf=zs.CONF,
            classes=watch or None,
            persist=True,
            verbose=False,
        )[0]
        zs.update_tracks(result, self.model, state)
        canvas = zs.draw_rois(frame)
        h = frame.shape[0]
        n_safe = n_danger = 0
        for tid, info in state.memory.items():
            fx, fy = zs.anchor_xy(info.xyxy, h)
            zname, zcolor = zs.zone_label(fx, fy, h)
            state.prev_zone[tid] = zname
            if zname == "SAFE":
                n_safe += 1
            else:
                n_danger += 1
            zs.draw_track(canvas, tid, info, zname, zcolor)
        if banner.active(video_time):
            zs.draw_alert_banner(canvas, banner.text)
        zs.draw_hud(canvas, n_safe, n_danger, state.alert_count)
        return canvas


ANALYZER = MultiCamAnalyzer()


def zone_to_text(zone: Optional[np.ndarray]) -> str:
    if zone is None:
        return ""
    pts = np.asarray(zone, dtype=int)
    return "; ".join(f"{x},{y}" for x, y in pts)


def text_to_zone(text: str) -> np.ndarray:
    pts = []
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        x_s, y_s = part.split(",")
        pts.append([int(float(x_s)), int(float(y_s))])
    if len(pts) < 3:
        raise ValueError("폴리곤은 점 3개 이상 필요합니다. 예: 40,300; 180,230; 420,200")
    return np.array(pts, dtype=np.int32)


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLS)


def load_events() -> pd.DataFrame:
    if not EVENT_CSV.exists():
        return empty_events()
    try:
        df = pd.read_csv(EVENT_CSV, encoding="utf-8-sig")
    except Exception:
        return empty_events()
    for col in EVENT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[EVENT_COLS].copy()


def load_inbox() -> str:
    if not ALERT_INBOX.exists():
        return "(관제 알림함 비어 있음 — zoo_safety.py 실행 후 알림이 쌓입니다)"
    text = ALERT_INBOX.read_text(encoding="utf-8").strip()
    return text if text else "(관제 알림함 비어 있음)"


def latest_snapshots(n: int = SNAPSHOT_LIMIT) -> List[str]:
    if not SNAP_DIR.exists():
        return []
    files = sorted(
        SNAP_DIR.glob("alert_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [str(p) for p in files[:n]]


def high_count(df: pd.DataFrame) -> int:
    if len(df) == 0:
        return 0
    return int((df["severity"].astype(str).str.upper() == "HIGH").sum())


def count_kpi(df: pd.DataFrame) -> str:
    total = len(df)
    last = str(df.iloc[-1]["detected_at"]) if total else "-"
    cams = df["camera_id"].nunique() if total else 0
    return (
        f"### 총 이벤트 **{total}**건\n"
        f"- HIGH 심각도: **{high_count(df)}**\n"
        f"- 카메라 수: **{cams}**\n"
        f"- 최근 시각: `{last}`"
    )


def connected_cameras() -> List[dict]:
    return [c for c in CAMERAS if c["file"].exists()]


def camera_status(df: pd.DataFrame) -> str:
    status = "🔴 ALERT" if high_count(df) else "🟢 NORMAL"
    lines = [
        "### 카메라 상태",
        f"- 연결: **{len(connected_cameras())}/{len(CAMERAS)}**채널",
        f"- 상태: **{status}**",
        f"- 누적 알림: **{len(df)}**건",
    ]
    for cam in CAMERAS:
        badge = "🟢" if cam["file"].exists() else "⚫"
        lines.append(
            f"- {badge} `{cam['code']}` {cam['title']} · `{cam['file'].name}`"
        )
    return "\n".join(lines)


def alert_banner(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return (
            '<div class="alert-panel ok">'
            '<span class="alert-icon" aria-hidden="true">✓</span>'
            "<div>"
            "<h3>정상 운영 중</h3>"
            "<p>위험 이벤트가 없습니다. 동물과 관람객 안전을 계속 감시합니다.</p>"
            "</div></div>"
        )
    last = df.iloc[-1]
    return (
        '<div class="alert-panel">'
        '<span class="alert-icon" aria-hidden="true">!</span>'
        "<div>"
        f"<h3>[{last.get('detected_at', '')}] "
        f"{last.get('camera_id', CAMERA_ID)} · "
        f"{last.get('event_type', '-')} "
        f"({str(last.get('severity', '')).upper()})</h3>"
        f"<p>{last.get('message', '')}</p>"
        "</div></div>"
    )


def group_chart(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame({label: ["(없음)"], "건수": [0]})
    return (
        df.groupby(col, dropna=False)
        .size()
        .reset_index(name="건수")
        .rename(columns={col: label})
    )


def events_table(df: pd.DataFrame, keyword: str = "") -> pd.DataFrame:
    view = df.copy()
    if keyword and keyword.strip():
        kw = keyword.strip().lower()
        mask = view.astype(str).apply(
            lambda col: col.str.lower().str.contains(kw, na=False)
        ).any(axis=1)
        view = view[mask]
    view = view.tail(TABLE_LIMIT).iloc[::-1].reset_index(drop=True)
    return view.rename(columns=DISPLAY_COLS)[list(DISPLAY_COLS.values())]


def refresh_all(keyword: str = "") -> Tuple:
    df = load_events()
    kpi = count_kpi(df)
    cam = camera_status(df)
    inbox = load_inbox()
    return (
        alert_banner(df),
        kpi,
        kpi,
        cam,
        cam,
        group_chart(df, "event_type", "유형"),
        group_chart(df, "severity", "심각도"),
        events_table(df, keyword),
        inbox,
        inbox,
        latest_snapshots(),
    )


def refresh_detect() -> List[Optional[np.ndarray]]:
    try:
        return ANALYZER.step_all()
    except Exception as e:
        print("[detect]", e)
        return [None] * len(CAMERAS)


def apply_roi(cam_label: str, text: str) -> Tuple[str, Optional[np.ndarray]]:
    idx = next(i for i, c in enumerate(CAMERAS) if c["code"] == cam_label)
    cam = CAMERAS[idx]
    try:
        zone = text_to_zone(text)
        cam["safe_zone"] = zone
        msg = f"✅ `{cam['code']}` ROI 적용 ({len(zone)}점)"
    except Exception as e:
        return f"❌ ROI 오류: {e}", None

    ANALYZER._ensure()
    frames = ANALYZER.step_all()
    return msg, frames[idx]


def load_roi_text(cam_label: str) -> str:
    cam = next(c for c in CAMERAS if c["code"] == cam_label)
    ANALYZER._ensure()
    zone = cam.get("safe_zone")
    if zone is None and cam["file"].exists():
        cap = cv2.VideoCapture(str(cam["file"]))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
        cap.release()
        zone = full_frame_safe_zone(w, h)
        cam["safe_zone"] = zone
    return zone_to_text(zone)


# ---------------------------------------------------------------------------
# 정적 HTML · 테마
# ---------------------------------------------------------------------------
def load_css() -> str:
    extra = """
video {
  width: 100% !important;
  background: #000;
}
.detect-grid img {
  border-radius: 8px;
  border: 1px solid #ffffff14;
}
"""
    base = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    return base + extra


def header_html() -> str:
    return """
<div class="sz-header">
  <a class="sz-logo" href="#">SAFEZOO</a>
  <nav class="sz-nav" aria-label="주 메뉴">
    <a class="active" href="#home">홈</a>
    <a href="#cameras">카메라</a>
    <a href="#alerts">안전 알림</a>
  </nav>
  <span class="sz-demo-label">프로토타입 · Zoo Vision</span>
</div>
"""


def hero_html() -> str:
    return """
<section class="sz-hero" id="home">
  <div class="sz-hero-content">
    <span class="sz-eyebrow">SAFEZOO CONTROL CENTER</span>
    <h1>모든 생명을 위한<br />안전한 시선.</h1>
    <p>
      동물과 관람객의 위험을 발견하고,<br />
      알림부터 안전조치까지 한 화면에서 확인하세요.
    </p>
    <div class="sz-actions">
      <a class="sz-btn sz-btn-primary" href="#cameras">카메라 보기</a>
      <a class="sz-btn sz-btn-secondary" href="#alerts">안전 알림 보기</a>
    </div>
  </div>
</section>
"""


def camera_card_html(cam: dict) -> str:
    connected = cam["file"].exists()
    badge_cls = "sz-badge live" if connected else "sz-badge"
    badge_text = "연결됨" if connected else "영상 미연결"
    return f"""
<article class="sz-camera-card">
  <div class="sz-camera-preview">
    <span class="{badge_cls}">{badge_text}</span>
    {cam["code"]}
  </div>
  <div class="sz-camera-info">
    <h3>{cam["title"]}</h3>
    <p>{cam["desc"]}</p>
  </div>
</article>
"""


def cameras_html() -> str:
    cards = "".join(camera_card_html(cam) for cam in CAMERAS)
    return f"""
<section class="sz-section" id="cameras">
  <h2>관제 카메라</h2>
  <div class="sz-camera-grid">{cards}</div>
</section>
"""


def footer_html() -> str:
    names = " · ".join(c["file"].stem for c in CAMERAS)
    return f"""
<footer class="sz-footer">
  SAFEZOO · Zoo Vision · {names}
</footer>
"""


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Noto Sans KR"), "Pretendard", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#141414",
        body_background_fill_dark="#141414",
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        block_background_fill="#232323",
        block_background_fill_dark="#232323",
        block_border_color="#ffffff14",
        block_border_color_dark="#ffffff14",
        block_label_text_color="#b3b3b3",
        block_title_text_color="#ffffff",
        border_color_primary="#ffffff22",
        button_primary_background_fill="#ffffff",
        button_primary_text_color="#141414",
        button_primary_background_fill_hover="#dddddd",
        input_background_fill="#232323",
        input_border_color="#ffffff22",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_live_tab():
    detect_imgs = []
    with gr.Tab("실시간 감시"):
        gr.Markdown("### 실시간 멀티뷰")
        with gr.Row(elem_classes=["detect-grid"]):
            for cam in CAMERAS[:2]:
                with gr.Column(min_width=280):
                    gr.Markdown(f"**{cam['code']} · {cam['title']}**")
                    detect_imgs.append(
                        gr.Image(
                            label=cam["file"].name,
                            height=280,
                            interactive=False,
                        )
                    )
        with gr.Row(elem_classes=["detect-grid"]):
            for cam in CAMERAS[2:]:
                with gr.Column(min_width=280):
                    gr.Markdown(f"**{cam['code']} · {cam['title']}**")
                    detect_imgs.append(
                        gr.Image(
                            label=cam["file"].name,
                            height=280,
                            interactive=False,
                        )
                    )

        with gr.Accordion("ROI 설정", open=False):
            cam_dd = gr.Dropdown(
                choices=[c["code"] for c in CAMERAS],
                value="CAMERA 02",
                label="카메라",
            )
            roi_text = gr.Textbox(
                label="SAFE 폴리곤 (x,y; x,y; …)",
                lines=3,
                value=zone_to_text(GIRAFFE_ZONE),
            )
            roi_btn = gr.Button("ROI 적용", variant="primary")
            roi_msg = gr.Markdown()
            roi_preview = gr.Image(label="미리보기", height=260, interactive=False)
            cam_dd.change(fn=load_roi_text, inputs=cam_dd, outputs=roi_text)
            roi_btn.click(
                fn=apply_roi,
                inputs=[cam_dd, roi_text],
                outputs=[roi_msg, roi_preview],
            )

        with gr.Row():
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("#### 최근 알림 스냅샷")
                gallery = gr.Gallery(
                    label="스냅샷",
                    columns=3,
                    height=220,
                    object_fit="contain",
                    preview=True,
                )
            with gr.Column(scale=1, min_width=250):
                cam_md = gr.Markdown(elem_classes=["panel-box"])
                live_kpi = gr.Markdown(elem_classes=["panel-box"])
                gr.Markdown("#### 관제센터 알림함")
                inbox_live = gr.Textbox(
                    lines=10,
                    max_lines=16,
                    interactive=False,
                    show_label=False,
                )
    return detect_imgs, gallery, cam_md, live_kpi, inbox_live


def build_stats_tab():
    with gr.Tab("통계"):
        with gr.Row():
            kpi = gr.Markdown(elem_classes=["panel-box"])
            cam_stat = gr.Markdown(elem_classes=["panel-box"])
        with gr.Row():
            chart_type = gr.BarPlot(
                x="유형", y="건수", title="이벤트 유형별 건수",
                x_title="유형", y_title="건수",
            )
            chart_sev = gr.BarPlot(
                x="심각도", y="건수", title="심각도별 건수",
                x_title="심각도", y_title="건수",
            )
    return kpi, cam_stat, chart_type, chart_sev


def build_events_tab():
    with gr.Tab("이벤트 기록"):
        with gr.Row():
            kw = gr.Textbox(
                label="검색 (유형·카메라·메시지·객체…)",
                placeholder="예: ZONE_EXIT / giraffe / HIGH",
                scale=4,
            )
            btn = gr.Button("새로고침", variant="primary", scale=1)
        table = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown("#### 관제센터 알림함 전문")
        inbox_full = gr.Textbox(lines=14, interactive=False, show_label=False)
    return kw, btn, table, inbox_full


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="SafeZoo 안전관제",
        theme=build_theme(),
        css=load_css(),
    ) as demo:
        gr.HTML(header_html())
        gr.HTML(hero_html())
        banner = gr.HTML(elem_id="alerts")
        gr.HTML(cameras_html())

        with gr.Tabs(elem_classes=["dark-tabs"]):
            detect_imgs, gallery, cam_md, live_kpi, inbox_live = build_live_tab()
            kpi, cam_stat, chart_type, chart_sev = build_stats_tab()
            kw, btn, table, inbox_full = build_events_tab()

        gr.HTML(footer_html())

        outputs = [
            banner,
            live_kpi,
            kpi,
            cam_md,
            cam_stat,
            chart_type,
            chart_sev,
            table,
            inbox_live,
            inbox_full,
            gallery,
        ]

        demo.load(fn=lambda: refresh_all(""), inputs=None, outputs=outputs)
        demo.load(fn=refresh_detect, inputs=None, outputs=detect_imgs)
        btn.click(fn=refresh_all, inputs=kw, outputs=outputs)
        kw.change(fn=refresh_all, inputs=kw, outputs=outputs)

        timer = gr.Timer(REFRESH_SEC)
        timer.tick(fn=refresh_all, inputs=kw, outputs=outputs)

        detect_timer = gr.Timer(DETECT_SEC)
        detect_timer.tick(fn=refresh_detect, inputs=None, outputs=detect_imgs)

    return demo


def main() -> None:
    print("SafeZoo · Zoo Vision 통합 관제보드")
    print("  이벤트 CSV :", EVENT_CSV)
    print("  관제 알림함:", ALERT_INBOX)
    print("  멀티뷰    :", f"{DETECT_SEC}초 주기 실시간 탐지")
    for cam in CAMERAS:
        ok = "OK" if cam["file"].exists() else "MISSING"
        print(f"  [{ok}] {cam['code']} {cam['title']} → {cam['file'].name}")
    build_app().launch()


if __name__ == "__main__":
    main()
