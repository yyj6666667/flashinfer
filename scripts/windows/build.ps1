# ==============================================================
#  FlashInfer Windows build script (invoked by build.bat).
#
#  Assumes build.bat has already activated MSVC and set CUDA_PATH.
#  Exit codes:
#    0  success
#    1  prerequisite missing
#    2  build failed
#    3  smoke test failed
# ==============================================================
[CmdletBinding()]
param(
    [switch]$SkipDeps,
    [switch]$SkipSmoke,
    [switch]$Clean,
    [string]$CudaArch = ""
)

$ErrorActionPreference = "Stop"
$script:StartTime = Get-Date

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "[OK]   $msg"    -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "[WARN] $msg"   -ForegroundColor Yellow }
function Write-Err2($msg)  { Write-Host "[ERR]  $msg"   -ForegroundColor Red }

# Parse plain-argv variants passed through from build.bat.
foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^--skip-deps$'  { $SkipDeps  = $true }
        '^--skip-smoke$' { $SkipSmoke = $true }
        '^--clean$'      { $Clean     = $true }
        '^--arch=(.+)$'  { $CudaArch  = $Matches[1] }
    }
}

# ---- locate repo root (two levels up from scripts/windows/) ----
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
Write-Step "Repository root: $RepoRoot"

# ---- sanity checks ----
Write-Step "Checking prerequisites"

try {
    $pyVer = (python --version 2>&1).Trim()
    Write-OK "Python: $pyVer"
} catch {
    Write-Err2 "python.exe not on PATH. Install Python 3.9-3.12 and add it to PATH."
    exit 1
}

try {
    $nvccVer = (nvcc --version 2>&1 | Select-String 'release' | ForEach-Object { $_.Line }).Trim()
    Write-OK "nvcc: $nvccVer"
} catch {
    Write-Err2 "nvcc not on PATH. Check CUDA_PATH and reopen the shell."
    exit 1
}

try {
    $clVer = (cl 2>&1 | Select-Object -First 1).ToString().Trim()
    Write-OK "cl.exe: $clVer"
} catch {
    Write-Err2 "cl.exe not on PATH. build.bat should have activated vcvars64 first."
    exit 1
}

# ---- submodules ----
Write-Step "Checking git submodules"
$cutlassHdr = Join-Path $RepoRoot '3rdparty\cutlass\include\cutlass\cutlass.h'
$spdlogHdr  = Join-Path $RepoRoot '3rdparty\spdlog\include\spdlog\spdlog.h'
if ((-not (Test-Path $cutlassHdr)) -or (-not (Test-Path $spdlogHdr))) {
    Write-Warn2 "Submodules missing; initializing..."
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { Write-Err2 "git submodule init failed"; exit 1 }
}
Write-OK "cutlass + spdlog present"

# ---- clear JIT cache if requested ----
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

# ---- pip dependencies ----
if (-not $SkipDeps) {
    Write-Step "Installing Python build prerequisites"
    python -m pip install --upgrade pip
    # ninja + apache-tvm-ffi are build/runtime requirements for JIT.
    # torch is left to the user because the correct wheel depends on the CUDA
    # major version. We check it below and guide the user if missing.
    python -m pip install ninja "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"
    if ($LASTEXITCODE -ne 0) { Write-Err2 "pip install failed"; exit 1 }
}

# ---- torch sanity ----
Write-Step "Verifying PyTorch + CUDA"
$torchCheck = @"
import sys
try:
    import torch
    print(f'torch={torch.__version__}', f'cuda={torch.version.cuda}', f'available={torch.cuda.is_available()}')
except ImportError:
    sys.exit(77)
"@
$torchOut = python -c $torchCheck 2>&1
if ($LASTEXITCODE -eq 77) {
    Write-Err2 "PyTorch is not installed."
    $cudaMajor = ($env:CUDA_PATH -split 'v')[-1] -split '\.' | Select-Object -First 1
    $idxUrl = switch ($cudaMajor) {
        '11' { 'https://download.pytorch.org/whl/cu118' }
        '12' { 'https://download.pytorch.org/whl/cu124' }
        '13' { 'https://download.pytorch.org/whl/cu130' }
        default { 'https://download.pytorch.org/whl/cu124' }
    }
    Write-Host "    Install the matching wheel, e.g.:"
    Write-Host "      pip install torch --index-url $idxUrl" -ForegroundColor Yellow
    exit 1
}
Write-OK $torchOut

# ---- set CUDA arch list ----
if ($CudaArch) {
    $env:FLASHINFER_CUDA_ARCH_LIST = $CudaArch
    Write-OK "FLASHINFER_CUDA_ARCH_LIST=$CudaArch"
} elseif (-not $env:FLASHINFER_CUDA_ARCH_LIST) {
    # Auto-detect from the first visible GPU; fall back to a safe default if none.
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

# ---- editable install ----
Write-Step "Building FlashInfer (pip install -e .)"
$env:FLASHINFER_JIT_VERBOSE = '1'   # surface compile errors early
python -m pip install --no-build-isolation -e . -v
if ($LASTEXITCODE -ne 0) {
    Write-Err2 "pip install failed. Inspect the log above for the first compile error."
    exit 2
}
Write-OK "flashinfer installed in editable mode"

# ---- smoke test ----
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
