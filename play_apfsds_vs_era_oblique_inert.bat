@echo off
REM Double-click to play the apfsds_vs_era_oblique_inert bake — the equal-areal-mass INERT
REM twin of play_apfsds_vs_era_oblique.bat (reactivity off, same geometry/mass/timing).
REM The A/B delta between the two isolates the reactive contribution (docs/PHYSICS.md 3.2).
REM Paths resolve relative to this file (%~dp0), so the repo can live anywhere.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\apfsds_vs_era_oblique_inert"
if errorlevel 1 pause
