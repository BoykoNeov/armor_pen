@echo off
REM Double-click to play the apfsds_vs_nera bake (NON-explosive reactive interlayer —
REM the persistent-bulge branch) in the Godot viewer. Same geometry as
REM play_apfsds_vs_era.bat / _inert, so the three are an equal-areal-mass A/B family
REM differing only in the filler's response path (docs/PHYSICS.md 3.3).
REM Paths resolve relative to this file (%~dp0), so the repo can live anywhere.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\apfsds_vs_nera"
if errorlevel 1 pause
