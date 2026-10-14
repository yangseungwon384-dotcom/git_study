import os
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Windows + Anaconda OpenMP 중복 로딩 우회 (zoo_safety.py 기준)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import zoo_safety as zs

# ==========================================
# 1. Page Config & CSS Style
# ==========================================
st.set_page_config(
    page_title="SafeZoo - YOLO + TypeSafe Jev Dashboard",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    
""", unsafe_allow_html=True)

# ==========================================
# 2. Config & Models Loading
# ==========================================
BASE = Path(__file__).resolve().parent

CAMERAS = {
    "기린 관람 구역 (CCTV-01)": {
        "file": BASE / "giraffe_cctv.mp4",
        "cam_id": "ZOO_CAM_GIRAFFE_01",
        "watch": ("giraffe", "person"),
        "roi": zs.SAFE_ZONE
    },
    "코끼리 방사장 (CCTV-02)": {
        "file": BASE / "elephant_cctv.mp4",
        "cam_id": "ZOO_CAM_ELEPHANT_01",
        "watch": ("elephant", "person"),
        "roi": np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)
    },
    "표범 방사장 (CCTV-03)": {
        "file": BASE / "Leopard_cctv.mp4",
        "cam_id": "ZOO_CAM_LEOPARD_01",
        "watch": ("cat", "dog", "bear", "person"),
        "roi": np.array([[50, 50], [600, 50], [600, 350], [50, 350]], dtype=np.int32)
    }
}

@st.cache_resource
def load_yolo_model():
    # zoo_safety.py 의 모델 로딩 로직 사용
    model, _ = zs.load_model()
    return model

# ==========================================
# 3. YOLO -> Jev State & Jev System One Client
# ==========================================
def convert_yolo_to_jev_state(frame_id: int, boxes, roi_polygon: np.ndarray, model_names: dict) -> Dict[str, Any]:
    """
    YOLOv8/26 감지 결과를 Jev 전송용 표준 JSON State로 변환
    """
    persons = []
    animals = []
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            name = model_names[cls_id]
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            center = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
            
            if name == "person":
                persons.append(center)
            else:
                animals.append(center)

    state = {
        "frame_id": frame_id,
        "summary": {"persons": len(persons), "animals": len(animals)},
        "risk_context": {
            "person_in_danger_roi": False,
            "min_person_to_animal_m": 99.0
        }
    }

    # ROI 침범 여부 및 객체 간 거리 계산
    for p_center in persons:
        # 안전구역(SAFE_ZONE) 밖으로 나갔는지 판정 (In-poly 실패 시 DANGER 영역)
        if not zs.in_poly(p_center[0], p_center[1], roi_polygon):
            state["risk_context"]["person_in_danger_roi"] = True
        
        for a_center in animals:
            dist = math.sqrt((p_center[0] - a_center[0])**2 + (p_center[1] - a_center[1])**2) * 0.02
            if dist < state["risk_context"]["min_person_to_animal_m"]:
                state["risk_context"]["min_person_to_animal_m"] = round(dist, 2)

    return state

def query_jev_model(state_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    TypeSafe Jev (System One) 빠른 결정 추론 Mock/API
    """
    ctx = state_json["risk_context"]
    
    # 1. ROI 탈출/위험구역 침범 (최고 위험)
    if ctx["person_in_danger_roi"]:
        return {
            "risk_type": {"choice": "VISITOR_INTRUSION", "confidence": 0.96},
            "risk_level": {"score": 9.2},
            "trigger_alarm": {"value": True, "probability": 0.97}
        }
    # 2. 동물 근접 (주의)
    elif ctx["min_person_to_animal_m"] <= 3.0:
        return {
            "risk_type": {"choice": "ANIMAL_APPROACH", "confidence": 0.82},
            "risk_level": {"score": 6.5},
            "trigger_alarm": {"value": True, "probability": 0.75}
        }
    # 3. 정상 상태
    else:
        return {
            "risk_type": {"choice": "NORMAL", "confidence": 0.99},
            "risk_level": {"score": 1.0},
            "trigger_alarm": {"value": False, "probability": 0.01}
        }

# ==========================================
# 4. UI Layout Definition
# ==========================================
st.title("🦁 SafeZoo 관제 센터 (YOLO + TypeSafe Jev)")
st.caption("실시간 YOLOv8/26 바운딩 박스 추적 및 TypeSafe AI Jev(System One) 연동 모니터링 대시보드")

st.sidebar.header("⚙️ 관제 카메라 및 설정")
selected_cam_name = st.sidebar.selectbox("카메라 선택", list(CAMERAS.keys()))
fps_speed = st.sidebar.slider("시뮬레이션 FPS", 1, 30, 10)
run_simulation = st.sidebar.button("▶ 영상 시뮬레이션 시작", type="primary")

cam_info = CAMERAS[selected_cam_name]

col_left, col_right = st.columns([6, 5])

with col_left:
    st.subheader("📹 실시간 CCTV 영상 (YOLO Overlaid)")
    video_placeholder = st.empty()
    
    st.subheader("📄 Jev 전달용 State JSON")
    json_placeholder = st.empty()

with col_right:
    st.subheader("⚡ TypeSafe AI Jev (System One) 결정 수령")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    metric_type = m_col1.empty()
    metric_score = m_col2.empty()
    metric_alarm = m_col3.empty()
    
    st.markdown("---")
    alert_placeholder = st.empty()
    
    st.subheader("📈 실시간 Risk Score 트렌드")
    chart_placeholder = st.empty()

# ==========================================
# 5. Main Loop
# ==========================================
if run_simulation:
    model = load_yolo_model()
    
    if not cam_info["file"].exists():
        st.error(f"비디오 파일을 찾을 수 없습니다: {cam_info['file'].name}")
    else:
        cap = cv2.VideoCapture(str(cam_info["file"]))
        frame_id = 0
        risk_history = []
        
        # 감시 대상 클래스 ID 추출
        watch_ids = [i for i, name in model.names.items() if name in cam_info["watch"]]
        
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id += 1
            
            # 1) YOLO Tracking 추론
            results = model.track(frame, conf=zs.CONF, classes=watch_ids or None, persist=True, verbose=False)[0]
            
            # 2) YOLO -> Jev State 변환
            jev_state = convert_yolo_to_jev_state(frame_id, results.boxes, cam_info["roi"], model.names)
            
            # 3) Jev 모델 추론
            jev_res = query_jev_model(jev_state)
            
            risk_score = jev_res["risk_level"]["score"]
            risk_type = jev_res["risk_type"]["choice"]
            alarm_prob = jev_res["trigger_alarm"]["probability"]
            risk_history.append({"Frame": frame_id, "Risk Score": risk_score})

            # 4) ROI 및 YOLO 추적 결과 캔버스 그리기 (zoo_safety.py 모듈 함수 활용)
            canvas = zs.draw_rois(frame)
            if results.boxes is not None:
                for box in results.boxes:
                    cls_name = model.names[int(box.cls[0])]
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    
                    # 상태에 따른 Bounding Box 색상 전환
                    color = (0, 255, 0) if risk_type == "NORMAL" else (0, 0, 255)
                    cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(canvas, f"{cls_name} {conf:.2f}", (xyxy[0], max(20, xyxy[1] - 8)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 5) UI 갱신
            video_placeholder.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)
            json_placeholder.json(jev_state)

            metric_type.metric("위험 유형 (Choice)", risk_type, f"Conf: {jev_res['risk_type']['confidence']*100:.0f}%")
            metric_score.metric("위험 점수 (Score)", f"{risk_score} / 10.0")
            metric_alarm.metric("알람 여부 (Noul)", "🚨 YES" if jev_res['trigger_alarm']['value'] else "🟢 NO", f"Prob: {alarm_prob*100:.0f}%")

            # 위험 레벨에 따른 경고 창 업데이트
            if risk_score >= 8.0:
                alert_placeholder.markdown(
                    f'<div style="padding:12px 16px;border-radius:8px;background:#3b0d0c;color:#ffb4ab;font-weight:600;">'
                    f'🚨 긴급 — {risk_type} · 점수 {risk_score} / 10.0 · 알람 확률 {alarm_prob * 100:.0f}%</div>',
                    unsafe_allow_html=True,
                )
            elif risk_score >= 5.0:
                alert_placeholder.markdown(
                    f'<div style="padding:12px 16px;border-radius:8px;background:#3d2e00;color:#ffe08a;font-weight:600;">'
                    f'⚠️ 주의 — {risk_type} · 점수 {risk_score} / 10.0 · 알람 확률 {alarm_prob * 100:.0f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.markdown(
                    f'<div style="padding:12px 16px;border-radius:8px;background:#0d3320;color:#b7f7c8;font-weight:600;">'
                    f'✅ 정상 — {risk_type} · 점수 {risk_score} / 10.0</div>',
                    unsafe_allow_html=True,
                )

            chart_placeholder.line_chart(pd.DataFrame(risk_history).set_index("Frame"))
            time.sleep(1.0 / fps_speed)

        cap.release()