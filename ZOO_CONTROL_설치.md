# ZOO-CONTROL(최종) 설치와 실행

이 브랜치 하나만 받으면 개발 PC와 같은 영상, 같은 안전구역(ROI), 같은 분석 모델로 관제 화면이 뜬다.

## 들어 있는 것

| 위치 | 내용 |
| --- | --- |
| `twin_control/` | 관제 서버와 화면 (v15) |
| `zoo_safe/` | 분석 엔진 |
| `zoo_vision.py`, `zoo_safety.py`, `safezoo.css` | 엔진이 읽는 원본 카메라 설정과 구역 판정 코드 |
| `elephant_cctv.mp4`, `giraffe_cctv.mp4`, `tiger_cctv.mp4`, `redpanda_cctv.mp4` | 카메라 4대 영상 |
| `yolo26n.pt` | YOLO 탐지 모델 |
| `twin_control/roi_seed.json` | 카메라 4대의 안전구역. 처음 켤 때 자동으로 채워진다 |
| `twin_control/map/` | 3D 지도 설정과 안내도 이미지 |
| `requirements.txt` | 필요한 파이썬 패키지와 버전 |

## 들어 있지 않은 것

- 사건 기록과 로그인 계정(`twin_control/twin_data/`): 개인 기록이라 올리지 않는다. 새 PC에서는 빈 상태로 시작한다.
- 판단 AI 모델: Ollama로 따로 받는다(아래 4번). 없어도 영상 분석과 경보는 동작한다.

## 설치 (Windows)

1. 받기 (폴더 이름은 자유)
   ```
   git clone -b "ZOO-CONTROL(최종)" https://github.com/yangseungwon384-dotcom/git_study.git zoo_control
   cd zoo_control
   ```
2. 파이썬 3.14 설치 후, 받은 폴더 안에 가상환경을 만든다 (반드시 이 이름과 위치: `zoo_control\.venv`)
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. 관리자 계정 만들기 (비밀번호는 창에서 직접 입력)
   ```
   .venv\Scripts\python twin_control\twin_server.py --add-user 이름 --role admin
   ```
4. (선택) 판단 AI: https://ollama.com 에서 Ollama를 설치하고 `ollama pull tev1:0.8b`
5. 실행: `twin_control\run_twin_control.bat` (이 PC만) 또는 `twin_control\run_twin_control_network.bat` (같은 내부망의 다른 PC에서도 접속)

처음 켜면 서버 창에 "기본 안전구역을 채웠습니다: …"가 나오고, `영상·원본 코드:`가 받은 폴더로 표시된다.

## 같은 화면이 나오는 원리

- 서버는 `twin_control` 바로 위 폴더에 `zoo_vision.py`와 `giraffe_cctv.mp4`가 있으면 그 폴더의 영상·원본 코드·`yolo26n.pt`를 쓴다.
- 사건 DB에 구역이 없는 카메라는 `roi_seed.json`의 구역으로 채운다. 관제 화면에서 구역을 고치면 그 PC의 DB에 저장되고, 그 뒤로는 고친 구역을 쓴다.
- 실행 파일(.bat)은 받은 폴더의 `.venv` 파이썬을 먼저 찾는다.

## 다르게 나올 수 있는 것

- 분석 속도는 PC 성능(CPU)에 따라 다르다. 느린 PC에서는 같은 장면이라도 경보가 열리는 시점이 조금 다를 수 있다.
- 패키지 버전을 바꾸면 YOLO 결과가 미세하게 달라질 수 있다. `requirements.txt` 버전 그대로 설치하는 것이 가장 정확하다.
- 3D 지도의 three.js는 인터넷(cdnjs)에서 불러온다. 인터넷이 없으면 2D 지도만 나온다.
