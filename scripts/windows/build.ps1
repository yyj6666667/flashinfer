# ==============================================================
#  FlashInfer Windows build script (invoked by build.bat).
#
#  Flow:
#    1. Inspect the environment and collect *every* missing prerequisite.
#       For each one, print a winget command (preferred) and a manual
#       download URL so the user can install everything in one pass.
#    2. If any prerequisite is missing, exit 1 before doing any work.
#    3. Otherwise run submodule init -> pip install -> smoke test.
#
#  The --check flag stops after step 1; useful for a fresh machine to
#  list what's missing without attempting to build.
#
#  Exit codes:
#    0  success
#    1  prerequisite missing or check failed
#    2  build failed
#    3  smoke test failed
# ==============================================================
[CmdletBinding()]
param(
    [switch]$SkipDeps,
    [switch]$SkipSmoke,
    [switch]$Clean,
    [switch]$Check,
    [string]$CudaArch = ""
)

$ErrorActionPreference = "Stop"
$script:StartTime = Get-Date

function Write-Step($msg)  { Write-Host "`n==> $msg"      -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "[OK]   $msg"    -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "[WARN] $msg"    -ForegroundColor Yellow }
function Write-Err2($msg)  { Write-Host "[ERR]  $msg"    -ForegroundColor Red }

# Parse plain-argv variants passed through from build.bat.
foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^--skip-deps$'  { $SkipDeps  = $true }
        '^--skip-smoke$' { $SkipSmoke = $true }
        '^--clean$'      { $Clean     = $true }
        '^--check$'      { $Check     = $true }
        '^--arch=(.+)$'  { $CudaArch  = $Matches[1] }
    }
}

# ==============================================================
# Missing-prerequisite collection
# ==============================================================
$script:Missing = @()

function Add-Missing {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Why,
        [string]$Winget = "",
        [string]$Manual = "",
        [string]$Notes  = ""
    )
    $script:Missing += [PSCustomObject]@{
        Name = $Name; Why = $Why; Winget = $Winget; Manual = $Manual; Notes = $Notes
    }
}

function Show-MissingSummary {
    if ($script:Missing.Count -eq 0) { return $false }
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Red
    Write-Err2 "$($script:Missing.Count) prerequisite(s) missing."
    Write-Host ("=" * 70) -ForegroundColor Red
    foreach ($m in $script:Missing) {
        Write-Host ""
        Write-Host ("-- {0} --" -f $m.Name) -ForegroundColor Yellow
        Write-Host "  Problem: $($m.Why)"
        if ($m.Winget) {
            Write-Host "  Install with winget (recommended):"
            Write-Host "    $($m.Winget)" -ForegroundColor Green
        }
        if ($m.Manual) {
            Write-Host "  Manual download:"
            Write-Host "    $($m.Manual)" -ForegroundColor Cyan
        }
        if ($m.Notes) {
            Write-Host "  Notes: $($m.Notes)"
        }
    }
    Write-Host ""
    Write-Host "After installing, open a NEW terminal (so PATH/CUDA_PATH refresh) and re-run:" -ForegroundColor Yellow
    Write-Host "    scripts\windows\build.bat" -ForegroundColor Cyan
    return $true
}

# ==============================================================
# Pre-check: is winget available? (not fatal, just for UX)
# ==============================================================
$hasWinget = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
if (-not $hasWinget) {
    Write-Warn2 "winget (Windows Package Manager) not found."
    Write-Warn2 "  Install 'App Installer' from Microsoft Store, or use the manual"
    Write-Warn2 "  download URLs below. See https://aka.ms/getwinget"
}

# ==============================================================
# Environment prerequisite checks
# ==============================================================
Write-Step "Checking prerequisites"

# ---- Git ----
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Add-Missing -Name "Git for Windows" `
        -Why "git.exe not on PATH (needed for submodule init and version stamping)" `
        -Winget "winget install -e --id Git.Git" `
        -Manual "https://git-scm.com/download/win"
} else {
    Write-OK ("git: " + (git --version).Trim())
}

# ---- Python ----
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Add-Missing -Name "Python 3.10 - 3.12" `
        -Why "python.exe not on PATH" `
        -Winget "winget install -e --id Python.Python.3.12" `
        -Manual "https://www.python.org/downloads/windows/" `
        -Notes "Check 'Add python.exe to PATH' during install. Python 3.13/3.14 are not yet supported by PyTorch on Windows."
} else {
    $pyVer = (python --version 2>&1).Trim()
    Write-OK "Python: $pyVer"
    try {
        $verStr = (python -c "import sys; print('{0}.{1}'.format(*sys.version_info[:2]))").Trim()
        $v = [Version]$verStr
        if ($v.Major -ne 3 -or $v.Minor -lt 9 -or $v.Minor -gt 12) {
            Write-Warn2 "Python $verStr detected; torch Windows wheels currently target 3.9-3.12."
            Write-Warn2 "  Consider installing Python 3.12 alongside your current version:"
            Write-Warn2 "    winget install -e --id Python.Python.3.12"
        }
    } catch {}
}

# ---- CUDA toolkit (nvcc) ----
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    if ($env:CUDA_PATH) {
        # CUDA_PATH set but bin not on PATH — try to recover transparently.
        $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    }
}
$nv = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nv) {
    $why = if ($env:CUDA_PATH) {
        "CUDA_PATH=$($env:CUDA_PATH) but nvcc.exe not found under its bin\ directory"
    } else {
        "nvcc.exe not on PATH and CUDA_PATH not set"
    }
    Add-Missing -Name "NVIDIA CUDA Toolkit 12.x" `
        -Why $why `
        -Winget "winget install -e --id Nvidia.CUDA" `
        -Manual "https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64" `
        -Notes "After install, close and reopen the shell so CUDA_PATH (set by the installer) takes effect. Choose a version matching your NVIDIA driver; run 'nvidia-smi' to see the max supported CUDA version."
} else {
    $nvccVer = (nvcc --version 2>&1 | Select-String 'release' | ForEach-Object { $_.Line }).Trim()
    Write-OK "nvcc: $nvccVer"
}

# ---- MSVC host compiler (cl.exe) ----
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    $wingetCmd = 'winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --add Microsoft.VisualStudio.Component.VC.CMake.Project"'
    Add-Missing -Name "Visual Studio 2022 Build Tools (MSVC x64)" `
        -Why "cl.exe not on PATH; MSVC is required as the host compiler for nvcc" `
        -Winget $wingetCmd `
        -Manual "https://visualstudio.microsoft.com/downloads/?q=build+tools" `
        -Notes "If you installed via the VS Installer UI, enable the 'Desktop development with C++' workload. After install, relaunch scripts\windows\build.bat — it will auto-activate vcvars64.bat."
} else {
    try {
        $clVer = (cl 2>&1 | Select-Object -First 1).ToString().Trim()
        Write-OK "cl.exe: $clVer"
    } catch {
        Write-OK "cl.exe: found (version output unavailable)"
    }
}

# ---- NVIDIA driver sanity (informational only) ----
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Write-Warn2 "nvidia-smi not on PATH. You likely do not have an NVIDIA GPU or driver."
    Write-Warn2 "  FlashInfer needs a CUDA-capable GPU at runtime."
    Write-Warn2 "  Download driver: https://www.nvidia.com/Download/index.aspx"
} else {
    $drvLine = (nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1 | Select-Object -First 1)
    Write-OK "GPU driver: $drvLine"
}

# Show all missing items and stop — installing components incrementally is
# slow; we'd rather the user install everything in one pass.
if (Show-MissingSummary) { exit 1 }

if ($Check -and -not $SkipDeps -and -not $SkipSmoke) {
    # --check mode: stop after environment checks + dependency checks below.
}

# ==============================================================
# Repo + submodule checks
# ==============================================================
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
Write-Step "Repository root: $RepoRoot"

Write-Step "Checking git submodules"
$cutlassHdr = Join-Path $RepoRoot '3rdparty\cutlass\include\cutlass\cutlass.h'
$spdlogHdr  = Join-Path $RepoRoot '3rdparty\spdlog\include\spdlog\spdlog.h'
if ((-not (Test-Path $cutlassHdr)) -or (-not (Test-Path $spdlogHdr))) {
    if ($Check) {
        Write-Warn2 "Submodules not initialized. Run:"
        Write-Warn2 "    git submodule update --init --recursive"
    } else {
        Write-Warn2 "Submodules missing; initializing..."
        git submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) {
            Write-Err2 "git submodule init failed"
            exit 1
        }
    }
} else {
    Write-OK "cutlass + spdlog present"
}

# ==============================================================
# Python package checks
# ==============================================================
if (-not $SkipDeps) {
    Write-Step "Checking Python build-time packages (ninja, apache-tvm-ffi)"
    $pyDeps = @"
missing = []
try:
    import ninja  # noqa
except ImportError:
    missing.append('ninja')
try:
    import tvm_ffi  # noqa
except ImportError:
    missing.append('apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2')
print(','.join(missing))
"@
    $missingPy = (python -c $pyDeps 2>&1).Trim()
    if ($missingPy) {
        if ($Check) {
            Write-Warn2 "Missing Python build packages: $missingPy"
            Write-Warn2 "Install with:"
            Write-Warn2 "    python -m pip install $($missingPy -replace ',', ' ')"
        } else {
            Write-Step "Installing missing Python build packages: $missingPy"
            python -m pip install --upgrade pip
            $args = $missingPy -split ','
            python -m pip install @args
            if ($LASTEXITCODE -ne 0) { Write-Err2 "pip install failed"; exit 1 }
        }
    } else {
        Write-OK "ninja + apache-tvm-ffi present"
    }
}

# ==============================================================
# Torch + CUDA sanity
# ==============================================================
Write-Step "Verifying PyTorch + CUDA"
$torchCheck = @"
import sys
try:
    import torch
    print('torch={}'.format(torch.__version__), 'cuda={}'.format(torch.version.cuda), 'available={}'.format(torch.cuda.is_available()))
except ImportError:
    sys.exit(77)
"@
$torchOut = python -c $torchCheck 2>&1
if ($LASTEXITCODE -eq 77) {
    # Resolve the best torch index URL based on detected CUDA major.minor.
    $cudaMajor = 0; $cudaMinor = 0
    if ($env:CUDA_PATH) {
        $m = [regex]::Match($env:CUDA_PATH, 'v(\d+)\.(\d+)')
        if ($m.Success) {
            $cudaMajor = [int]$m.Groups[1].Value
            $cudaMinor = [int]$m.Groups[2].Value
        }
    }
    $idxUrl = switch ($cudaMajor) {
        11 { 'https://download.pytorch.org/whl/cu118' }
        12 { if ($cudaMinor -ge 6) { 'https://download.pytorch.org/whl/cu126' } else { 'https://download.pytorch.org/whl/cu124' } }
        13 { 'https://download.pytorch.org/whl/cu130' }
        default { 'https://download.pytorch.org/whl/cu124' }
    }

    Write-Err2 "PyTorch is not installed."
    Write-Host ""
    Write-Host "-- PyTorch --" -ForegroundColor Yellow
    Write-Host "  Install the CUDA-matching Windows wheel:"
    Write-Host "    pip install torch --index-url $idxUrl" -ForegroundColor Green
    Write-Host "  See https://pytorch.org/get-started/locally/ to pick a variant manually."
    exit 1
}
Write-OK $torchOut

# Verify torch actually sees CUDA (common failure: driver too old for torch's CUDA runtime).
$cudaOk = (python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1) -eq $null
if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "torch imported but torch.cuda.is_available() is False."
    Write-Warn2 "  Common causes:"
    Write-Warn2 "    1. NVIDIA driver older than the torch CUDA runtime."
    Write-Warn2 "    2. torch CPU-only wheel installed (reinstall with --index-url above)."
    Write-Warn2 "  Check driver: nvidia-smi"
}

# ==============================================================
# Stop here if --check
# ==============================================================
if ($Check) {
    Write-Host ""
    Write-OK "All prerequisites present. Run without --check to build."
    exit 0
}

# ==============================================================
# CUDA architecture list
# ==============================================================
if ($CudaArch) {
    $env:FLASHINFER_CUDA_ARCH_LIST = $CudaArch
    Write-OK "FLASHINFER_CUDA_ARCH_LIST=$CudaArch"
} elseif (-not $env:FLASHINFER_CUDA_ARCH_LIST) {
    $detect = @"
import torch, sys
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    suffix = 'a' if major >= 9 else ''
    print(f'{major}.{minor}{suffix}')
else:
    sys.exit(1)
"@
    try {
        $arch = (python -c $detect).Trim()
        $env:FLASHINFER_CUDA_ARCH_LIST = $arch
        Write-OK "Auto-detected FLASHINFER_CUDA_ARCH_LIST=$arch"
    } catch {
        $env:FLASHINFER_CUDA_ARCH_LIST = '8.9'
        Write-Warn2 "Could not auto-detect GPU; defaulting to FLASHINFER_CUDA_ARCH_LIST=8.9"
    }
}

# ==============================================================
# Optional: clear JIT cache
# ==============================================================
if ($Clean) {
    Write-Step "Clearing JIT cache"
    $cache = Join-Path $env:USERPROFILE '.cache\flashinfer'
    if (Test-Path $cache) {
        Remove-Item -Recurse -Force $cache
        Write-OK "Removed $cache"
    } else {
        Write-OK "No cache at $cache (already clean)"
    }
}

# ==============================================================
# Editable install
# ==============================================================
Write-Step "Building FlashInfer (pip install -e .)"
$env:FLASHINFER_JIT_VERBOSE = '1'
python -m pip install --no-build-isolation -e . -v
if ($LASTEXITCODE -ne 0) {
    Write-Err2 "pip install failed. Inspect the log above for the first compile error."
    exit 2
}
Write-OK "flashinfer installed in editable mode"

# ==============================================================
# Smoke test
# ==============================================================
if (-not $SkipSmoke) {
    Write-Step "Running smoke test"
    python "$PSScriptRoot\smoke_test.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Err2 "Smoke test failed (exit $LASTEXITCODE)."
        exit 3
    }
    Write-OK "Smoke test passed"
}

$elapsed = (Get-Date) - $script:StartTime
Write-Host ("`nDone in {0:mm\:ss}." -f $elapsed) -ForegroundColor Green
exit 0
