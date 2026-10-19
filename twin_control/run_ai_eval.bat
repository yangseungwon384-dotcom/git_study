@echo off
chcp 65001 >nul
title ZOO-GIS Twin Control - AI 채점
setlocal

rem 사람이 판정한 사건을 TEV에 다시 묻고, 얼마나 맞는지 채점한다. Ollama가 켜져 있어야 한다.
rem 사건별 결과는 이 폴더의 ai_eval_result.csv 에 저장된다.

rem 파이썬: 저장소 안의 .venv → 이 PC의 기존 .venv → PATH의 python 순서로 찾는다.
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\code\code_test\zoo\.venv\Scripts\python.exe"
if not exist "%PY%" for %%P in (python.exe) do set "PY=%%~$PATH:P"
set PYTHONIOENCODING=utf-8

if not exist "%PY%" (
  echo 파이썬 가상환경을 찾을 수 없습니다: %PY%
  pause
  exit /b 1
)

"%PY%" "%~dp0twin_eval.py" --ask --out "%~dp0ai_eval_result.csv" %*

echo.
pause
