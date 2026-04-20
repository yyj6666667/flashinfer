# Resilient git pull for Windows behind Clash Verge (GFW).
# If the pull times out at github.com, restart Clash Verge and retry once.
#
#   .\scripts\windows\pull_resilient.ps1              # pulls origin windows
#   .\scripts\windows\pull_resilient.ps1 main          # pulls origin main
[CmdletBinding()]
param([string]$Branch = "windows")

function Restart-ClashVerge {
    $candidates = @(
        'Clash Verge', 'clash-verge', 'clash-verge-service',
        'verge-mihomo', 'clash-meta'
    )
    $procs = Get-Process -Name $candidates -ErrorAction SilentlyContinue
    if (-not $procs) {
        Write-Warning "[pull] No Clash Verge process found to restart"
        return
    }
    $paths = $procs | ForEach-Object { $_.Path } | Where-Object { $_ } | Select-Object -Unique
    $procs | ForEach-Object {
        Write-Host "[pull] stopping $($_.ProcessName) ($($_.Id))" -ForegroundColor Yellow
        try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch {}
    }
    Start-Sleep -Seconds 2
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Write-Host "[pull] restarting $p" -ForegroundColor Yellow
            Start-Process -FilePath $p
        }
    }
    Start-Sleep -Seconds 15  # give the proxy time to re-establish upstream
}

$maxAttempts = 4
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    Write-Host "[pull] attempt ${attempt}/${maxAttempts}: git pull origin $Branch" -ForegroundColor Cyan
    git pull origin $Branch
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[pull] success" -ForegroundColor Green
        exit 0
    }
    Write-Warning "[pull] git pull failed (exit $LASTEXITCODE)"
    if ($attempt -lt $maxAttempts) {
        if ($attempt -eq 1) {
            # First retry: just wait a bit, proxy might self-heal
            Start-Sleep -Seconds 5
        } else {
            Restart-ClashVerge
        }
    }
}
Write-Error "[pull] git pull failed after $maxAttempts attempts; giving up"
exit 1
