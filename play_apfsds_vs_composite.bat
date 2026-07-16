@echo off
REM Double-click to play the apfsds_vs_composite bake (bonded RHA/ceramic/RHA stack) in the Godot viewer.
REM Paths resolve relative to this file (%~dp0), so the repo can live anywhere.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\apfsds_vs_composite"
if errorlevel 1 pause
