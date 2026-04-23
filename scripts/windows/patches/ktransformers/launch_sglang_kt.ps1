$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "Continue"

# vcvars env (cl.exe for any JIT)
$envDump = "C:\flashinfer\.sglang_launch_logs\vcvars_env.txt"
Get-Content $envDump | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2] }
}
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:SGLANG_LOGGING_LEVEL = "info"

# hwloc DLL so kt_kernel can load
$env:PATH = "C:\hwloc\bin;$env:PATH"

$logDir = "C:\flashinfer\.sglang_launch_logs"
$stdout  = Join-Path $logDir "kt_stdout.log"
$stderr  = Join-Path $logDir "kt_stderr.log"
$pidFile = Join-Path $logDir "kt_launch.pid"
"" | Set-Content $stdout; "" | Set-Content $stderr

$py = "C:\flashinfer\.venv\Scripts\python.exe"
$modelPath = "C:/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
$sglangArgs = @(
    "-u", "-m", "sglang.launch_server",
    "--model-path", $modelPath,
    "--served-model-name", "Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4-KT",
    "--port", "30000", "--host", "127.0.0.1",
    "--trust-remote-code",
    "--dtype", "float16",
    "--kt-cpuinfer", "20",
    "--kt-method", "GPTQ_INT4",
    "--kt-num-gpu-experts", "0",
    "--kt-weight-path", $modelPath,
    "--kt-threadpool-count", "1",
    "--kt-numa-nodes", "0",
    "--mem-fraction-static", "0.7",
    "--max-total-tokens", "2048",
    "--chunked-prefill-size", "512",
    "--disable-cuda-graph"
)
Write-Output ("args: " + ($sglangArgs -join " "))

$proc = Start-Process -FilePath $py -ArgumentList $sglangArgs `
    -WorkingDirectory "C:\sglang" `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -NoNewWindow -PassThru
$proc.Id | Out-File -FilePath $pidFile -Encoding ascii
Write-Output ("launch pid=" + $proc.Id)

# Poll ready / generate / exit — in-session
$deadline = (Get-Date).AddSeconds(600)
$started = $false
$generated = $false
$dead = $false
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { $dead = $true; break }
    if (-not $started) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:30000/model_info" -TimeoutSec 3
            Write-Output ("MODEL_INFO: " + ($r | ConvertTo-Json -Compress))
            $started = $true
        } catch { Start-Sleep -Seconds 6; continue }
    }
    if ($started -and -not $generated) {
        try {
            $body = '{"text": "The three primary colors are", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}'
            $gen = Invoke-RestMethod -Uri "http://127.0.0.1:30000/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 360
            Write-Output ("GENERATE: " + ($gen | ConvertTo-Json -Compress))
            $generated = $true; break
        } catch {
            Write-Output ("generate error: " + $_.Exception.Message)
            if ($proc.HasExited) { $dead = $true; break }
            Start-Sleep -Seconds 8
        }
    }
}

if (-not $generated -and -not $dead) { Write-Output "TIMEOUT" }
if ($dead) { Write-Output ("PROCESS EXITED exitcode=" + $proc.ExitCode) }
Write-Output ("--- stdout tail ---")
Get-Content $stdout -Tail 50
Write-Output ("--- stderr tail ---")
Get-Content $stderr -Tail 80

if ($generated) { Write-Output ("KEEP_ALIVE pid=" + $proc.Id) }
else {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Write-Output "KILLED"
}
