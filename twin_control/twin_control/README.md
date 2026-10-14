# ZOO-GIS Twin Control

`zoo_safe`의 분석 엔진을 그대로 돌리고, 그 결과를 한 화면짜리 관제 대시보드로 보여 주는 서버다. Gradio 화면은 띄우지 않는다. 엔진 코드는 고치지 않고 가져다 쓴다.

지도 위치, 생체 수치, 잠금 상태, 무전은 모의 데이터다. 영상 검출과 사건만 실제 분석 결과다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `twin_server.py` | FastAPI 서버. 영상(MJPEG), 상태(WebSocket), 사건·구역·AI 판단 API, 로그인 |
| `twin_vision.py` | 카메라 표와 탐지 후처리 (종 묶기, 겹침 제거, 정지 물체 제외, 한글 라벨) |
| `twin_control.html` | 관제 화면. 서버 없이 열면 모의 데이터만으로 동작한다 |
| `run_twin_control.bat` | 이 PC에서만 실행 |
| `run_twin_control_network.bat` | 같은 네트워크의 다른 PC에서도 접속하게 실행 (계정 필요) |

## 실행

`zoo_safe`와 같은 가상환경을 쓴다. `fastapi`, `uvicorn`, `websockets`, `pillow`가 더 필요하다.

```powershell
python twin_control\twin_server.py            # http://127.0.0.1:8765
python twin_control\twin_server.py --ollama   # 이 PC의 Ollama TEV로 실시간 AI 판단
```

영상과 `zoo_vision.py`는 `--source-dir`(기본 `C:\code\code_test\zoo`)에서 읽는다. 카메라 표는 `twin_vision.py`의 `CAMERAS`에 있고, 지금은 `elephant_cctv.mp4`, `giraffe_cctv.mp4`, `tiger_cctv.mp4`, `redpanda_cctv.mp4`를 쓴다. `tiger_cctv.mp4`와 `redpanda_cctv.mp4`는 이 저장소에 없으므로 따로 넣어야 한다.

배치 파일 두 개에는 가상환경 경로가 `C:\code\code_test\zoo\.venv`로 적혀 있다. 다른 PC에서는 그 줄을 고친다.

## 화면

- **통합 관제**: 동물 카드, 지도, CCTV 4칸, 이벤트 피드, 타임라인, 모의 시나리오 재생기
- **카메라**: 영상만 크게 보기, 1분할, 안전구역 편집
- **CSV 기록**: 하루치 사건 표, 필터, CSV 내려받기, AI 판단 일괄 요청

## 다른 PC에서 접속

```powershell
python twin_control\twin_server.py --add-user 이름 --role admin   # 비밀번호는 입력창에서 받는다
python twin_control\twin_server.py --host 0.0.0.0
```

계정이 하나라도 있으면 모든 화면과 영상이 로그인을 요구한다. 계정 없이 `--host`를 바꾸면 실행을 거부한다. `admin`은 구역 편집까지, `operator`는 사건 확인·검토와 AI 판단 요청까지 할 수 있다.

HTTPS는 없다. 내부망에서만 쓰고 인터넷에 직접 열지 않는다.

## 알아 둘 점

- 사건 DB, 계정, 증거 사진은 `twin_control/twin_data/`에 생기고 저장소에는 올리지 않는다.
- 기본 YOLO에는 호랑이와 레서판다가 없다. 다른 클래스로 잡힌 것을 카메라별로 묶어 "호랑이 추정", "레서판다 추정"으로 표시할 뿐이라, 놓치는 프레임이 있다.
- `twin_vision.py`의 기준값은 시험 영상 4개로 정했다. 현장 카메라에서는 다시 재야 한다.
- AI 판단(TEV)은 참고 표시다. 사건의 확인 상태와 규칙 경고를 바꾸지 않는다. 시험에서는 1순위 확률이 0.27~0.41로 낮았다.
