@echo off
chcp 65001 >nul
title ZOO-GIS Twin Control (Cloudflare Tunnel)
setlocal

rem 인터넷의 다른 PC에서 접속할 수 있는 임시 주소(https://....trycloudflare.com)를 만든다.
rem 로그인 계정이 있어야 열린다. 이 창을 닫으면 외부 주소가 끊긴다.

set "PY=C:\code\code_test\zoo\.venv\Scripts\python.exe"
set "SERVER=%~dp0twin_server.py"
set "CF=C:\Program Files (x86)\cloudflared\cloudflared.exe"
set "URL=http://127.0.0.1:8765"

if not exist "%PY%" (
  echo 파이썬 가상환경을 찾을 수 없습니다: %PY%
  pause
  exit /b 1
)
if not exist "%CF%" set "CF=cloudflared"

set "CODE=000"
for /f %%c in ('curl -s -o nul -m 3 -w "%%{http_code}" %URL%/api/me') do set "CODE=%%c"

if "%CODE%"=="200" (
  echo 지금 켜진 서버는 로그인 없이 열려 있습니다. 이대로 공개하면 누구나 들어옵니다.
  echo 그 서버 창을 닫은 뒤 이 파일을 다시 실행하세요.
  pause
  exit /b 1
)
if "%CODE%"=="401" goto tunnel

echo 등록된 계정:
"%PY%" "%SERVER%" --list-users
echo.
set "NEWUSER="
set /p "NEWUSER=새 관리자 계정을 만들려면 이름을 입력하세요 (건너뛰려면 Enter): "
if defined NEWUSER "%PY%" "%SERVER%" --add-user "%NEWUSER%" --role admin

echo.
echo 관제 서버를 새 창에서 시작합니다. 계정이 없으면 그 창에 안내가 나오고 서버는 켜지지 않습니다.
start "ZOO-GIS Twin Control server" cmd /k ""%PY%" "%SERVER%" --require-login --ollama"

echo 서버가 준비될 때까지 기다립니다 (최대 90초).
set /a TRIES=0
:wait
set "CODE=000"
for /f %%c in ('curl -s -o nul -m 3 -w "%%{http_code}" %URL%/api/me') do set "CODE=%%c"
if "%CODE%"=="401" goto tunnel
set /a TRIES+=1
if %TRIES% geq 30 (
  echo 서버가 켜지지 않았습니다. 새로 뜬 서버 창의 안내를 확인하세요.
  pause
  exit /b 1
)
timeout /t 3 /nobreak >nul
goto wait

:tunnel
echo.
echo 터널을 엽니다. 아래에 나오는 https://....trycloudflare.com 주소를 공유하세요.
echo 주소는 켤 때마다 바뀝니다. 이 창을 닫으면 외부 접속이 끊깁니다. 서버는 서버 창을 닫아야 꺼집니다.
echo.
"%CF%" tunnel --url %URL%

echo.
echo 터널이 닫혔습니다.
pause
