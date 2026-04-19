# Sweep RMSNorm (and family) perf on Windows using the repo's own
# benchmarks/flashinfer_benchmark.py runner. Used to validate that the
# Windows port preserves Linux-level performance — in particular the
# __launch_bounds__(1024) cap added to RMSNormKernel should not cost
# meaningful throughput on bandwidth-bound norm workloads.
#
# Output:
#   .debug/perf/<routine>_<dtype>_<stamp>.csv   <- per-row results (append)
#   .debug/perf/<routine>_<dtype>_<stamp>.log   <- full stdout+stderr
#
# Usage:
#   scripts\windows\perf\run_norm_perf.ps1                       # full sweep
#   scripts\windows\perf\run_norm_perf.ps1 -Quick                # 2x2 smoke
#   scripts\windows\perf\run_norm_perf.ps1 -Routine rmsnorm_quant
#   scripts\windows\perf\run_norm_perf.ps1 -Dtype bfloat16
[CmdletBinding()]
param(
    [string]$Routine = 'rmsnorm',
    [string]$Dtype   = 'float16',
    [int]$NumIters   = 100,
    [switch]$Quick
)

$ErrorActionPreference = 'Continue'

# MSVC + CUDA must be on PATH for the first JIT compile; activate.ps1
# is idempotent so this is cheap on warm shells.
& (Join-Path $PSScriptRoot '..\activate.ps1')

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$PerfDir  = Join-Path $RepoRoot '.debug\perf'
New-Item -ItemType Directory -Force -Path $PerfDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$tag   = "${Routine}_${Dtype}_${stamp}"
$csv   = Join-Path $PerfDir "$tag.csv"
$log   = Join-Path $PerfDir "$tag.log"

# Shapes mirror common LLaMA-style dims. 8192 is the one where the
# launch_bounds fix kicked in, so keep it in the grid.
if ($Quick) {
    $batches     = @(1, 99)
    $hiddenSizes = @(4096, 8192)
} else {
    $batches     = @(1, 19, 99, 989)
    $hiddenSizes = @(1024, 3072, 4096, 8192, 16384)
}

$bench = Join-Path $RepoRoot 'benchmarks\flashinfer_benchmark.py'
if (-not (Test-Path $bench)) {
    Write-Error "[perf] $bench not found. Run from the repo root or fix path."
    exit 1
}

# Capture identifying context into the log header so CSVs from Linux /
# Windows / different commits are self-describing.
$header = @(
    "# perf sweep",
    "# timestamp:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "# host:       $env:COMPUTERNAME",
    "# os:         Windows",
    "# routine:    $Routine",
    "# dtype:      $Dtype",
    "# num_iters:  $NumIters",
    "# commit:     $(git -C $RepoRoot rev-parse --short HEAD 2>$null)",
    "# branch:     $(git -C $RepoRoot rev-parse --abbrev-ref HEAD 2>$null)"
) -join "`n"
$header | Out-File -FilePath $log -Encoding utf8

Push-Location $RepoRoot
try {
    $total = $batches.Count * $hiddenSizes.Count
    $i = 0
    foreach ($b in $batches) {
        foreach ($h in $hiddenSizes) {
            $i++
            Write-Host ("[perf {0}/{1}] batch={2} hidden={3}" -f $i, $total, $b, $h) `
                -ForegroundColor Cyan
            # --use_cuda_events: bench_gpu_time falls back to CUDA events.
            # CUPTI is the default but cupti-python only ships wheels for
            # CUDA >= 13, so on our CUDA 12.6 Windows box the CUPTI path
            # is unavailable. Accuracy difference is typically <2% which
            # is well within shot-to-shot variance for norm kernels.
            python $bench `
                --routine $Routine `
                --batch_size $b `
                --hidden_size $h `
                --input_dtype $Dtype `
                --backends cuda `
                --num_iters $NumIters `
                --use_cuda_events `
                --output_path $csv 2>&1 |
                Tee-Object -Append -FilePath $log
        }
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[perf] CSV: $csv" -ForegroundColor Green
Write-Host "[perf] Log: $log" -ForegroundColor Green
Write-Host "[perf] Tip: rerun the same command on Linux to produce a"
Write-Host "            comparable CSV for the port regression check."
