# Bisect which nvcc flag or flag-group triggers
#
#     nvcc fatal : A single input file is required for a non-link phase
#                  when an outputfile is specified
#
# on this Windows machine. Starts from a minimal command and adds one
# group of flags at a time until the failure reproduces; stops on the
# first failing group and records which one it was.
#
# Writes the full trace to .debug/last_run.log (UTF-8, no BOM) and
# pushes it so the Linux diagnostic session can pull and analyse.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir   = Join-Path $RepoRoot '.debug'
$LogFile  = Join-Path $LogDir 'last_run.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$lines = New-Object System.Collections.Generic.List[string]
function Log($s) { $lines.Add($s); Write-Host $s }

# --- Sandbox: minimal .cu that does NOT pull any FlashInfer / cutlass headers ---
$TmpDir = Join-Path $env:TEMP 'fi_nvcc_diag'
Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
$SrcFile = Join-Path $TmpDir 't.cu'
$ObjFile = Join-Path $TmpDir 't.cuda.obj'
@"
#include <cuda_runtime.h>
__global__ void k() {}
int main() { return 0; }
"@ | Set-Content -Encoding ascii $SrcFile

# --- Resolve nvcc the same way the JIT layer does ---
$nvccCmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
$nvcc = if ($nvccCmd) { $nvccCmd.Source } else { $null }
if (-not $nvcc -and $env:CUDA_PATH) {
    $nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
}
if (-not $nvcc -or -not (Test-Path $nvcc)) {
    Log "ERROR: nvcc not found (checked PATH and CUDA_PATH)"
    [System.IO.File]::WriteAllLines($LogFile, $lines, (New-Object System.Text.UTF8Encoding $false))
    exit 1
}

Log "Using nvcc: $nvcc"
Log "Using src:  $SrcFile"
Log "Using obj:  $ObjFile"
Log "--- nvcc --version ---"
(& $nvcc --version 2>&1 | Out-String).TrimEnd().Split("`n") | ForEach-Object { Log $_ }

# --- Flag groups, CUMULATIVE. Order mirrors cpp_ext.py emission order. ---
# Each subsequent group inherits all prior flags. A failure at group X
# therefore means: baseline(A..X-1) was fine, adding X broke it.
$groups = [ordered]@{
    'A_minimal'        = @()
    'B_std'            = @('-std=c++17')
    'C_arch'           = @('-gencode=arch=compute_89,code=sm_89')
    'D_threads'        = @('--threads=1')
    'E_fastmath'       = @('-use_fast_math')
    'F_relaxed'        = @('--expt-relaxed-constexpr')
    'G_xcompiler_host' = @('-Xcompiler=/EHsc', '-Xcompiler=/MD', '-Xcompiler=/bigobj')
    'H_defines'        = @(
        '-DFLASHINFER_ENABLE_FP8_E8M0',
        '-DFLASHINFER_ENABLE_FP4_E2M1',
        '-DFLASHINFER_ENABLE_F16',
        '-DFLASHINFER_ENABLE_BF16',
        '-DFLASHINFER_ENABLE_FP8_E4M3',
        '-DFLASHINFER_ENABLE_FP8_E5M2',
        '-DNDEBUG',
        '-DENABLE_BF16',
        '-DENABLE_FP8'
    )
    'I_opt'            = @('-O3')
    'J_debug_block'    = @(
        '-g', '-O0', '-G', '-lineinfo',
        '-Xptxas=-v',
        '-DCUTLASS_DEBUG_TRACE_LEVEL=2'
    )
    'K_showIncludes'   = @('-Xcompiler=/showIncludes')
}

$cum = @()
$firstFail = $null
foreach ($key in $groups.Keys) {
    $cum += $groups[$key]
    $argv = @($cum + @('-c', $SrcFile, '-o', $ObjFile))
    Log ""
    Log "=== Try: $key ==="
    Log ("  argv: " + ($argv -join ' '))
    $out = & $nvcc @argv 2>&1
    $ec  = $LASTEXITCODE
    if ($out) {
        (($out | Out-String).TrimEnd()).Split("`n") | ForEach-Object { Log ("    " + $_) }
    }
    Log "  => exit=$ec"
    if ($ec -ne 0 -and -not $firstFail) {
        $firstFail = $key
        Log ""
        Log "*** FIRST FAILING GROUP: $key ***"
        Log ("*** Flags added in this group: " + ($groups[$key] -join ' ') + " ***")
        break
    }
}

if (-not $firstFail) {
    Log ""
    Log "*** ALL GROUPS PASSED (bug did not reproduce on minimal source) ***"
    Log "    Flag set is innocent; the failure must be in something else."
    Log "    Next: replay the full flag set but with an output path whose"
    Log "    directory components contain dots (mirrors 0.6.7 in real cache)."
    Log ""

    $dotDir = Join-Path $TmpDir '0.6.7\89\cached_ops\norm'
    New-Item -ItemType Directory -Force -Path $dotDir | Out-Null
    $dotObj = Join-Path $dotDir 't.cuda.obj'
    $argv = @($cum + @('-c', $SrcFile, '-o', $dotObj))

    Log "=== Try: L_dots_in_outpath ==="
    Log ("  argv: " + ($argv -join ' '))
    $out = & $nvcc @argv 2>&1
    $ec  = $LASTEXITCODE
    if ($out) {
        (($out | Out-String).TrimEnd()).Split("`n") | ForEach-Object { Log ("    " + $_) }
    }
    Log "  => exit=$ec"
    if ($ec -ne 0) {
        $firstFail = 'L_dots_in_outpath'
        Log ""
        Log "*** CONFIRMED: nvcc chokes on output path whose directory segment"
        Log "    contains dots (e.g. '0.6.7'). This is the real root cause."
    } else {
        Log ""
        Log "*** L_dots_in_outpath PASSED. Root cause is elsewhere (maybe"
        Log "    specific to the real norm.cu source). Next hypothesis needed."
    }
}

# --- Write UTF-8 without BOM so Linux tooling reads it directly ---
[System.IO.File]::WriteAllLines($LogFile, $lines, (New-Object System.Text.UTF8Encoding $false))

Push-Location $RepoRoot
try {
    git add .debug/last_run.log | Out-Null
    git commit -m "diag: nvcc flag bisect $(Get-Date -Format 'yyyy-MM-dd HH:mm') first_fail=$firstFail" --allow-empty | Out-Null
    git push origin HEAD
} finally {
    Pop-Location
}

Write-Host ""
if ($firstFail) {
    Write-Host "Pushed diagnostic log. First failing group: $firstFail" -ForegroundColor Yellow
} else {
    Write-Host "Pushed diagnostic log. No group failed on minimal source." -ForegroundColor Cyan
}
