@echo off
chcp 65001 >nul
title ZOO-GIS Twin Control (network)
setlocal

rem 같은 네트워크의 다른 PC에서도 접속할 수 있게 서버를 켠다. 로그인 계정이 있어야 한다.
rem 계정이 없으면 관리자 계정부터 만든다. 비밀번호는 이 창에서 직접 입력한다.

rem 파이썬: 저장소 안의 .venv → 이 PC의 기존 .venv → PATH의 python 순서로 찾는다.
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\code\code_test\zoo\.venv\Scripts\python.exe"
if not exist "%PY%" for %%P in (python.exe) do set "PY=%%~$PATH:P"
set "SERVER=%~dp0twin_server.py"

if not exist "%PY%" (
  echo 파이썬 가상환경을 찾을 수 없습니다: %PY%
  pause
  exit /b 1
)

echo 등록된 계정:
"%PY%" "%SERVER%" --list-users
echo.
set "NEWUSER="
set /p "NEWUSER=새 관리자 계정을 만들려면 이름을 입력하세요 (건너뛰려면 Enter): "
if defined NEWUSER "%PY%" "%SERVER%" --add-user "%NEWUSER%" --role admin

echo.
echo 다른 PC에서는 아래 IPv4 주소 뒤에 :8765 를 붙여 접속합니다. 예: http://192.168.0.10:8765
ipconfig | findstr /c:"IPv4"
echo.
echo 관제 서버를 시작합니다. 이 창을 닫으면 서버도 꺼집니다.
echo Windows 방화벽이 물으면 "개인 네트워크"만 허용하세요.
echo.

"%PY%" "%SERVER%" --host 0.0.0.0 --ollama %*

echo.
echo 서버가 꺼졌습니다. 위에 오류가 있으면 확인하세요.
pause
