@echo off
REM ==============================================================
REM  FlashInfer - Windows quick build/install entry point
REM
REM  Usage (from any shell):
REM      scripts\windows\build.bat                     REM full install + smoke test
REM      scripts\windows\build.bat --check             REM only check prerequisites
REM      scripts\windows\build.bat --skip-deps         REM skip pip-installing deps
REM      scripts\windows\build.bat --skip-smoke        REM build only, no kernel test
REM      scripts\windows\build.bat --clean             REM clear JIT cache first
REM
REM  This wrapper best-effort activates the MSVC x64 toolchain via
REM  vcvars64.bat, then hands off to build.ps1 which is the single
REM  source of truth for prerequisite checks and install hints.
REM
REM  If MSVC or CUDA are missing, the wrapper does NOT exit early;
REM  it still runs build.ps1 so the user sees a complete diagnostic
REM  (with winget commands and download URLs) for everything at once.
REM ==============================================================
setlocal EnableDelayedExpansion

REM -------- skip vcvars activation if cl.exe is already on PATH --------
where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] cl.exe already on PATH, skipping vcvars activation.
    goto :run_ps
)

REM -------- locate vswhere --------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [WARN] vswhere.exe not found at "%VSWHERE%".
    echo        Visual Studio 2022 Build Tools may not be installed.
    echo        Continuing; build.ps1 will print install instructions.
    goto :run_ps
)

REM -------- find latest VS install with C++ x64 tools --------
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)
if not defined VS_PATH (
    echo [WARN] No Visual Studio installation with C++ x64 tools found.
    echo        Continuing; build.ps1 will print install instructions.
    goto :run_ps
)

set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [WARN] vcvars64.bat not found under "%VS_PATH%".
    goto :run_ps
)

echo [INFO] Activating MSVC environment from "%VCVARS%"
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [WARN] vcvars64.bat activation failed; continuing anyway.
)

:run_ps
REM -------- prepend CUDA bin to PATH if CUDA_PATH is set --------
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
)

REM -------- hand off to PowerShell for the real work --------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
exit /b %errorlevel%
