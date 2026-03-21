@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_full_env.ps1" %*
exit /b %ERRORLEVEL%
