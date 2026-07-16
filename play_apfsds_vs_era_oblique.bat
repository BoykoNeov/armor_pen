@echo off
REM Double-click to play the apfsds_vs_era_oblique bake (reactive sandwich, 55 deg obliquity)
REM in the Godot viewer. Compare against play_apfsds_vs_era_oblique_inert.bat — the equal-
REM areal-mass inert twin — to see the reactive contribution (docs/PHYSICS.md 3.2).
REM Paths resolve relative to this file (%~dp0), so the repo can live anywhere.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\apfsds_vs_era_oblique"
if errorlevel 1 pause
