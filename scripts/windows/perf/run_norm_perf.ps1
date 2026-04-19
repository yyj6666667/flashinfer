# Sweep rmsnorm perf via benchmarks/flashinfer_benchmark.py.
# --use_cuda_events because cupti-python has no Windows wheel for CUDA 12.x.
[CmdletBinding()]
param(
    [string]$Routine = 'rmsnorm',
    [string]$Dtype   = 'float16',
    [int]$NumIters   = 100,
    [int[]]$Batches  = @(1, 99, 989),
    [int[]]$Hiddens  = @(1024, 4096, 8192, 16384)
)

& (Join-Path $PSScriptRoot '..\activate.ps1')

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Csv  = Join-Path $Repo ".debug\perf\$Routine-$Dtype.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $Csv) | Out-Null
Remove-Item -Force $Csv -ErrorAction SilentlyContinue

Push-Location $Repo
try {
    foreach ($b in $Batches) { foreach ($h in $Hiddens) {
        Write-Host "batch=$b hidden=$h" -ForegroundColor Cyan
        python benchmarks\flashinfer_benchmark.py --routine $Routine `
            --batch_size $b --hidden_size $h --input_dtype $Dtype `
            --backends cuda --num_iters $NumIters --use_cuda_events `
            --output_path $Csv
    }}
} finally { Pop-Location }

Write-Host "`nCSV: $Csv" -ForegroundColor Green
