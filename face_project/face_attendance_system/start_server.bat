@echo off
cd /d "%~dp0"
start /B python run.py > nul 2>&1
echo Server started on http://127.0.0.1:5000
