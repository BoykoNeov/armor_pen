@echo off
REM Double-click for a menu: pick a scenario + color mode, then launch the viewer.
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"

echo ============================================
echo   armor_pen viewer -- pick a scenario
echo ============================================
echo   1) apfsds_vs_era        (reactive ERA sandwich)
echo   2) apfsds_vs_era_inert  (inert control)
echo   3) apfsds_vs_rha        (plain RHA plate)
echo.
set "CACHE="
set /p choice="Scenario number (1-3): "
if "%choice%"=="1" set "CACHE=apfsds_vs_era"
if "%choice%"=="2" set "CACHE=apfsds_vs_era_inert"
if "%choice%"=="3" set "CACHE=apfsds_vs_rha"
if not defined CACHE (
  echo No valid choice - nothing to play.
  pause
  goto :eof
)

echo.
echo   Color by:  1) material   2) velocity
set /p c2="Color number (default 1): "
set "COLOR=material_id"
if "%c2%"=="2" set "COLOR=vel_mag"

echo.
echo Launching %CACHE%  (color: %COLOR%) ...
echo Controls:  Space=pause  Left/Right=step  Up/Down=speed  R=restart  Esc=quit
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\%CACHE%" --color %COLOR%
if errorlevel 1 pause
