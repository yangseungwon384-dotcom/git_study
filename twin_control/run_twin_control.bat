@echo off
chcp 65001 >nul
title ZOO-GIS Twin Control
setlocal

rem 두 번 눌러 실행한다. 이 PC에서만 열리는 관제 서버를 켜고 브라우저를 연다.
rem 다른 PC에서도 접속하게 하려면 run_twin_control_network.bat 을 쓴다.

set "PY=C:\code\code_test\zoo\.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8765"

if not exist "%PY%" (
  echo 파이썬 가상환경을 찾을 수 없습니다: %PY%
  pause
  exit /b 1
)

curl -s -o nul -m 2 %URL%/login
if not errorlevel 1 (
  echo 서버가 이미 켜져 있습니다. 브라우저만 엽니다.
  if not defined TWIN_NO_BROWSER start "" %URL%
  exit /b 0
)

echo 관제 서버를 시작합니다. 이 창을 닫으면 서버도 꺼집니다.
echo 10초쯤 뒤에 브라우저가 열립니다: %URL%
echo.
if not defined TWIN_NO_BROWSER start "" /min cmd /c "timeout /t 10 /nobreak >nul & start "" %URL%"

"%PY%" "%~dp0twin_server.py" --ollama %*

echo.
echo 서버가 꺼졌습니다. 위에 오류가 있으면 확인하세요.
pause
