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

# Null/array-safe version-string capture. Any output from the command (stdout
# or stderr) is coerced to a single trimmed string. Returns '' if the command
# produces nothing (e.g. the Microsoft Store App Execution Alias for python,
# which silently drops --version and triggers a NullReference on .Trim()).
#
# We locally override ErrorActionPreference to 'Continue' so that tools that
# legitimately write to stderr (e.g. cl.exe prints its banner on stderr) do
# not trigger the script-level Stop-on-error behaviour and blow away the
# output we are trying to collect.
function Invoke-Capture {
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments = @())
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $oldPsNative = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
        $oldPsNative = $Global:PSNativeCommandUseErrorActionPreference
        $Global:PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $out = & $Exe @Arguments 2>&1
    } catch {
        return ''
    } finally {
        $ErrorActionPreference = $oldEap
        if ($null -ne $oldPsNative) {
            $Global:PSNativeCommandUseErrorActionPreference = $oldPsNative
        }
    }
    if ($null -eq $out) { return '' }
    # Flatten arrays and coerce ErrorRecords / other objects to string.
    return (($out | ForEach-Object { "$_" }) -join "`n").Trim()
}

# A command is "really present" only if (a) Get-Command finds it AND (b) the
# executable actually produces output for the supplied probe arguments. This
# distinguishes real installs from WindowsApps Microsoft Store redirect stubs.
function Test-Usable {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string[]]$ProbeArgs = @('--version')
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    # Detect python/python3 WindowsApps aliases specifically — they live under
    # %LOCALAPPDATA%\Microsoft\WindowsApps and are usually the reason probe
    # commands return empty strings non-interactively.
    $src = if ($cmd.Source) { $cmd.Source } else { '' }
    $isStoreAlias = $src -match 'WindowsApps\\(python|python3)\.exe$'
    $ver = Invoke-Capture -Exe $Name -Arguments $ProbeArgs
    if ([string]::IsNullOrWhiteSpace($ver)) {
        if ($isStoreAlias) {
            return [PSCustomObject]@{ Version = ''; Source = $src; StoreAlias = $true }
        }
        return [PSCustomObject]@{ Version = ''; Source = $src; StoreAlias = $false }
    }
    return [PSCustomObject]@{ Version = $ver; Source = $src; StoreAlias = $false }
}

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
        [string]$Manual = "",
        [string]$Notes  = ""
    )
    $script:Missing += [PSCustomObject]@{
        Name = $Name; Why = $Why; Manual = $Manual; Notes = $Notes
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
        if ($m.Manual) {
            Write-Host "  Download/install:"
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
# Environment prerequisite checks
# ==============================================================
Write-Step "Checking prerequisites"

# ---- Git ----
$gitInfo = Test-Usable -Name 'git' -ProbeArgs @('--version')
if (-not $gitInfo -or [string]::IsNullOrWhiteSpace($gitInfo.Version)) {
    Add-Missing -Name "Git for Windows" `
        -Why "git.exe not on PATH (needed for submodule init and version stamping)" `
        -Manual "https://git-scm.com/download/win" `
        -Notes "Run the installer with defaults. Reopen the shell after install so PATH refreshes."
} else {
    Write-OK ("git: " + $gitInfo.Version)
}

# ---- Python ----
$pyInfo = Test-Usable -Name 'python' -ProbeArgs @('--version')
if ((-not $pyInfo) -or [string]::IsNullOrWhiteSpace($pyInfo.Version) -or $pyInfo.StoreAlias) {
    $why = if ($pyInfo -and $pyInfo.StoreAlias) {
        "python.exe on PATH resolves to the Microsoft Store App Execution Alias ($($pyInfo.Source)), which is a stub, not a real interpreter. Disable the alias under Settings -> Apps -> Advanced app settings -> App execution aliases, or install Python separately."
    } else {
        "python.exe not on PATH"
    }
    Add-Missing -Name "Python 3.10 - 3.12" `
        -Why $why `
        -Manual "https://www.python.org/downloads/release/python-3128/" `
        -Notes "Download the 'Windows installer (64-bit)' for Python 3.12.x. During install, enable 'Add python.exe to PATH'. Python 3.13/3.14 are not yet supported by PyTorch on Windows."
} else {
    Write-OK "Python: $($pyInfo.Version)"
    $verStr = Invoke-Capture -Exe 'python' -Arguments @('-c', "import sys; print('{0}.{1}'.format(*sys.version_info[:2]))")
    if ($verStr) {
        try {
            $v = [Version]$verStr
            if ($v.Major -ne 3 -or $v.Minor -lt 9 -or $v.Minor -gt 12) {
                Write-Warn2 "Python $verStr detected; torch Windows wheels currently target 3.9-3.12."
                Write-Warn2 "  Install Python 3.12 alongside your current version:"
                Write-Warn2 "    https://www.python.org/downloads/release/python-3128/"
            }
        } catch {
            Write-Warn2 "Could not parse Python version string: $verStr"
        }
    }
}

# ---- CUDA toolkit (nvcc) ----
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    if ($env:CUDA_PATH) {
        # CUDA_PATH set but bin not on PATH — try to recover transparently.
        $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    }
}
$nvInfo = Test-Usable -Name 'nvcc' -ProbeArgs @('--version')
if (-not $nvInfo -or [string]::IsNullOrWhiteSpace($nvInfo.Version)) {
    $why = if ($env:CUDA_PATH) {
        "CUDA_PATH=$($env:CUDA_PATH) but nvcc.exe not found under its bin\ directory"
    } else {
        "nvcc.exe not on PATH and CUDA_PATH not set"
    }
    Add-Missing -Name "NVIDIA CUDA Toolkit 12.x" `
        -Why $why `
        -Manual "CUDA 12.8 (recommended, matches torch cu128): https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=Server2022&target_type=exe_local" `
        -Notes "Pick CUDA 12.x (NOT 13.x). 12.8 has the broadest torch Windows wheel coverage (cu128). Alternatives: 12.6 (cu126), 12.4 (cu124). Archive: https://developer.nvidia.com/cuda-toolkit-archive . After running the .exe installer, CLOSE and REOPEN the shell so CUDA_PATH takes effect."
} else {
    $releaseLine = ($nvInfo.Version -split "`n" | Where-Object { $_ -match 'release' } | Select-Object -First 1)
    if (-not $releaseLine) { $releaseLine = ($nvInfo.Version -split "`n")[0] }
    Write-OK "nvcc: $($releaseLine.Trim())"
}

# ---- MSVC host compiler (cl.exe) ----
# cl.exe has no --version flag; running it bare prints a banner to stderr
# and then blocks on stdin. Since cl is not subject to App Execution
# Aliases, ``Get-Command`` pointing to a real .exe file is sufficient
# evidence that MSVC is installed and activated (via vcvars64.bat). This
# avoids a false negative where stderr output gets reinterpreted as a
# script-fatal error under PowerShell 7's strict native-command mode.
$clCmd = Get-Command cl -ErrorAction SilentlyContinue
$clPath = if ($clCmd) { $clCmd.Source } else { $null }
if ((-not $clPath) -or (-not (Test-Path $clPath))) {
    Add-Missing -Name "Visual Studio 2022 Build Tools (MSVC x64)" `
        -Why "cl.exe not on PATH; MSVC is required as the host compiler for nvcc" `
        -Manual "https://aka.ms/vs/17/release/vs_BuildTools.exe" `
        -Notes "Download the installer above, then run it with the C++ workload:`n           vs_BuildTools.exe --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`n         Alternatively, in the GUI installer enable 'Desktop development with C++'.`n         After install, relaunch scripts\windows\build.bat -- vcvars64.bat is auto-activated."
} else {
    Write-OK "cl.exe: $clPath"
}

# ---- NVIDIA driver sanity (informational only) ----
$smiInfo = Test-Usable -Name 'nvidia-smi' -ProbeArgs @('--query-gpu=driver_version,name', '--format=csv,noheader')
if (-not $smiInfo -or [string]::IsNullOrWhiteSpace($smiInfo.Version)) {
    Write-Warn2 "nvidia-smi not on PATH or returned no output."
    Write-Warn2 "  FlashInfer needs an NVIDIA GPU and driver at runtime."
    Write-Warn2 "  Download driver: https://www.nvidia.com/Download/index.aspx"
} else {
    $firstGpu = ($smiInfo.Version -split "`n")[0].Trim()
    Write-OK "GPU driver: $firstGpu"
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
        12 {
            if     ($cudaMinor -ge 8) { 'https://download.pytorch.org/whl/cu128' }
            elseif ($cudaMinor -ge 6) { 'https://download.pytorch.org/whl/cu126' }
            else                      { 'https://download.pytorch.org/whl/cu124' }
        }
        13 { 'https://download.pytorch.org/whl/cu130' }
        default { 'https://download.pytorch.org/whl/cu128' }
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
