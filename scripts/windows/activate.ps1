# Idempotent MSVC + CUDA environment activation for FlashInfer on Windows.
#
# What it does:
#   1. If cl.exe is not on PATH, locate vcvars64.bat via vswhere and import
#      its full environment block (PATH, INCLUDE, LIB, LIBPATH, ...) into
#      the current PowerShell session.
#   2. If nvcc.exe is not on PATH but CUDA_PATH is set, prepend
#      %CUDA_PATH%\bin so nvcc becomes reachable.
#   3. With -Clean, wipe ~/.cache/flashinfer to force a full JIT rebuild.
#
# Usage:
#   . .\scripts\windows\activate.ps1          # dot-source: affects current shell
#   . .\scripts\windows\activate.ps1 -Clean   # also clears JIT cache
#   .\scripts\windows\activate.ps1            # plain invoke: env changes only
#                                              propagate to child processes of
#                                              this script (useful when called
#                                              from another script).
[CmdletBinding()]
param([switch]$Clean, [switch]$Quiet)

function _log($msg, $color = 'DarkGray') {
    if (-not $Quiet) { Write-Host "[activate] $msg" -ForegroundColor $color }
}

# -------- Force UTF-8 code page for child processes --------
# On Chinese-locale Windows (CP936 / GBK):
#   - cmd's code page controls what child processes inherit (chcp 65001)
#   - PowerShell's .NET Console streams decode external-process bytes
#     using [Console]::OutputEncoding; default is OEM (GBK), which
#     mojibakes UTF-8 output from cl.exe / python into the log file.
cmd /c "chcp 65001 >nul 2>&1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# -------- MSVC (cl.exe) --------
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    _log "cl.exe already on PATH, skipping vcvars activation"
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Warning "[activate] vswhere.exe not found at $vswhere; install VS 2022 Build Tools"
    } else {
        $vsPath = & $vswhere -latest -products * `
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                    -property installationPath
        if (-not $vsPath) {
            Write-Warning "[activate] no VS 2022 installation with x64 VC tools found"
        } else {
            $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
            if (-not (Test-Path $vcvars)) {
                Write-Warning "[activate] vcvars64.bat not found at $vcvars"
            } else {
                cmd /c "`"$vcvars`" && set" | ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)$') {
                        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
                    }
                }
                $clPath = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
                _log "vcvars64 imported: $clPath" 'Green'
            }
        }
    }
}

# -------- CUDA (nvcc.exe) --------
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
        _log "added CUDA_PATH\bin to PATH: $env:CUDA_PATH\bin" 'Green'
    } else {
        Write-Warning "[activate] nvcc.exe not on PATH and CUDA_PATH not set or invalid"
    }
} else {
    _log "nvcc.exe already on PATH, skipping"
}

# -------- Optional cache wipe --------
# Use cmd's rd /s /q: dramatically faster than PowerShell's Remove-Item
# on large JIT trees (thousands of small files) and handles long paths
# more reliably than the .NET filesystem APIs Remove-Item wraps.
if ($Clean) {
    $cache = Join-Path $env:USERPROFILE '.cache\flashinfer'
    if (Test-Path $cache) {
        cmd /c "rd /s /q `"$cache`"" | Out-Null
        _log "cleared JIT cache: $cache" 'Yellow'
    } else {
        _log "no JIT cache to clear ($cache does not exist)"
    }
}
