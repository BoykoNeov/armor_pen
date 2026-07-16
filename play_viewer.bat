@echo off
REM Double-click for a menu: pick a scenario + color mode, then launch the viewer.
REM
REM Keep this list in sync with solver/scenarios/*.yaml — every deck that exists
REM should be bakeable AND listed here (root CLAUDE.md working conventions).
setlocal
set "GODOT=godot"
where godot >nul 2>nul || set "GODOT=%LOCALAPPDATA%\Microsoft\WinGet\Links\godot.exe"

echo ================================================================
echo   armor_pen viewer -- pick a scenario
echo ================================================================
echo   Kinetic rod vs a single stack:
echo     1) apfsds_vs_rha              plain 40 mm RHA plate
echo     2) apfsds_vs_composite        RHA / ceramic / RHA sandwich
echo     3) apfsds_vs_spaced           spaced plates with an air gap
echo.
echo   Reactive A/B pairs (play a deck against its _inert twin --
echo   the twins are equal-areal-mass controls, so the difference IS
echo   what the reactive layer contributes):
echo     4) apfsds_vs_era              reactive ERA sandwich, 0 deg
echo     5) apfsds_vs_era_inert          ^-- its inert control
echo     6) apfsds_vs_era_oblique      reactive ERA sandwich, 55 deg
echo     7) apfsds_vs_era_oblique_inert  ^-- its inert control
echo     8) apfsds_vs_nera             non-explosive bulging interlayer
echo.
echo   Shaped charge:
echo     9) heat_vs_composite          NOTE: jet is still a rod stand-in
echo.
set "CACHE="
set /p choice="Scenario number (1-9): "
if "%choice%"=="1" set "CACHE=apfsds_vs_rha"
if "%choice%"=="2" set "CACHE=apfsds_vs_composite"
if "%choice%"=="3" set "CACHE=apfsds_vs_spaced"
if "%choice%"=="4" set "CACHE=apfsds_vs_era"
if "%choice%"=="5" set "CACHE=apfsds_vs_era_inert"
if "%choice%"=="6" set "CACHE=apfsds_vs_era_oblique"
if "%choice%"=="7" set "CACHE=apfsds_vs_era_oblique_inert"
if "%choice%"=="8" set "CACHE=apfsds_vs_nera"
if "%choice%"=="9" set "CACHE=heat_vs_composite"
if not defined CACHE (
  echo No valid choice - nothing to play.
  pause
  goto :eof
)
if not exist "%~dp0caches\%CACHE%\manifest.json" (
  echo.
  echo   Cache "%CACHE%" is not baked yet. Bake it with:
  echo     cd solver
  echo     python -m ballistics_solver.run scenarios/%CACHE%.yaml --out ../caches/%CACHE%
  echo.
  pause
  goto :eof
)

echo.
echo   Color by:  1) material   2) velocity   3) damage   4) stress
set /p c2="Color number (default 1): "
set "COLOR=material_id"
if "%c2%"=="2" set "COLOR=vel_mag"
if "%c2%"=="3" set "COLOR=damage"
if "%c2%"=="4" set "COLOR=stress"

echo.
echo Launching %CACHE%  (color: %COLOR%) ...
echo Controls:  Space=pause  Left/Right=step  Up/Down=speed  C=cycle color
echo            Wheel or +/-=zoom  Drag(middle/right)=pan  F=fit  R=restart  Esc=quit
"%GODOT%" --path "%~dp0visualizer" -- --cache "%~dp0caches\%CACHE%" --color %COLOR%
if errorlevel 1 pause
