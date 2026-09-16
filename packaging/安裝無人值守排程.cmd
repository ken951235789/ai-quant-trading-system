@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0安裝無人值守排程.ps1"
if errorlevel 1 pause
