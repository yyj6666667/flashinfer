# Run a Python command (default: smoke_test.py), capture full stdout+stderr
# to .debug/last_run.log at the repo root, and push that log to the remote
# so the Linux-side Claude session can pull and diagnose without copy-paste.
#
# Usage:
#     scripts\windows\debug_run.ps1                  # runs smoke_test.py
#     scripts\windows\debug_run.ps1 pytest tests\... # runs arbitrary command
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Continue'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir   = Join-Path $RepoRoot '.debug'
$LogFile  = Join-Path $LogDir 'last_run.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $Command -or $Command.Count -eq 0) {
    $Command = @('python', (Join-Path $PSScriptRoot 'smoke_test.py'))
}

$header = @(
    "================================================================",
    "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "host:      $env:COMPUTERNAME",
    "cwd:       $RepoRoot",
    "cmd:       $($Command -join ' ')",
    "----------------------------------------------------------------"
) -join "`n"
$header | Out-File -FilePath $LogFile -Encoding utf8

Push-Location $RepoRoot
try {
    & $Command[0] $Command[1..($Command.Count - 1)] *>&1 | Tee-Object -Append -FilePath $LogFile
    $exit = $LASTEXITCODE
} finally {
    Pop-Location
}

"---- exit=$exit ----" | Out-File -Append -FilePath $LogFile -Encoding utf8

Push-Location $RepoRoot
try {
    git add .debug/last_run.log
    git commit -m "debug: run $(Get-Date -Format 'yyyy-MM-dd HH:mm') exit=$exit" --allow-empty | Out-Null
    git push origin HEAD
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Pushed .debug/last_run.log (exit=$exit). Tell the Linux session to pull." -ForegroundColor Cyan
exit $exit
