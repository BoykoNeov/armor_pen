@echo off
REM Double-click to play the apfsds_vs_era_inert bake (inert control) in the Godot viewer.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\apfsds_vs_era_inert"
if errorlevel 1 pause
