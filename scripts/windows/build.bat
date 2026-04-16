@echo off
REM ==============================================================
REM  FlashInfer - Windows quick build/install entry point
REM
REM  Usage (from any shell):
REM      scripts\windows\build.bat                     REM full install + smoke test
REM      scripts\windows\build.bat --skip-deps         REM skip pip-installing deps
REM      scripts\windows\build.bat --skip-smoke        REM build only, no kernel test
REM      scripts\windows\build.bat --clean             REM clear JIT cache first
REM
REM  This wrapper locates Visual Studio 2022 Build Tools, activates
REM  vcvars64.bat so cl.exe/link.exe are on PATH, then invokes the
REM  PowerShell build script that does the real work.
REM ==============================================================
setlocal EnableDelayedExpansion

REM -------- locate vswhere --------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found at "%VSWHERE%".
    echo         Install "Visual Studio 2022 Build Tools" with the
    echo         "Desktop development with C++" workload.
    exit /b 1
)

REM -------- find latest VS install with C++ x64 tools --------
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)
if not defined VS_PATH (
    echo [ERROR] No Visual Studio installation with the C++ x64 toolset was found.
    echo         Run "Visual Studio Installer" and add "MSVC v143 - VS 2022 C++ x64/x86 build tools".
    exit /b 1
)

set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat not found under "%VS_PATH%".
    exit /b 1
)

echo [INFO] Activating MSVC environment from "%VCVARS%"
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed.
    exit /b 1
)

REM -------- check CUDA_PATH (set by the CUDA Toolkit installer) --------
if not defined CUDA_PATH (
    echo [ERROR] CUDA_PATH is not set. Install the NVIDIA CUDA Toolkit or set CUDA_PATH
    echo         to your CUDA install directory, e.g.:
    echo           set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
    exit /b 1
)
echo [INFO] CUDA_PATH=%CUDA_PATH%

REM -------- ensure CUDA bin is on PATH for nvcc.exe discovery --------
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM -------- hand off to PowerShell --------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
exit /b %errorlevel%
