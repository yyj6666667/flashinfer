# Run a Python command (default: smoke_test.py), capture full stdout+stderr
# to .debug/last_run.log at the repo root, and push that log to the remote
# so the Linux-side Claude session can pull and diagnose without copy-paste.
#
# Usage:
#     scripts\windows\debug_run.ps1                  # runs smoke_test.py
#     scripts\windows\debug_run.ps1 -Clean           # wipe JIT cache first
#     scripts\windows\debug_run.ps1 pytest tests\... # runs arbitrary command
#
# The log is written as UTF-8 without BOM so `git diff` shows readable
# text, not "Bin X -> Y bytes". (Tee-Object would default to UTF-16 LE
# on PowerShell 5.x, which has embedded null bytes and makes git treat
# the file as binary.)
[CmdletBinding()]
param(
    [switch]$Clean,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Continue'

# Auto-activate MSVC + CUDA so the Python subprocess sees cl.exe / nvcc.
& (Join-Path $PSScriptRoot 'activate.ps1') -Clean:$Clean

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir   = Join-Path $RepoRoot '.debug'
$LogFile  = Join-Path $LogDir 'last_run.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $Command -or $Command.Count -eq 0) {
    $Command = @('python', (Join-Path $PSScriptRoot 'smoke_test.py'))
}

# Collect every output line in memory so we can flush once as UTF-8 no-BOM.
$lines = New-Object System.Collections.Generic.List[string]
function Log($s) { $lines.Add($s); Write-Host $s }

Log "================================================================"
Log "timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Log "host:      $env:COMPUTERNAME"
Log "cwd:       $RepoRoot"
Log "cmd:       $($Command -join ' ')"
Log "----------------------------------------------------------------"

Push-Location $RepoRoot
try {
    & $Command[0] $Command[1..($Command.Count - 1)] *>&1 | ForEach-Object {
        # $_ may be a string or an ErrorRecord; coerce to string and echo.
        $line = "$_"
        Log $line
    }
    $exit = $LASTEXITCODE
} finally {
    Pop-Location
}

Log "---- exit=$exit ----"

[System.IO.File]::WriteAllLines(
    $LogFile, $lines, (New-Object System.Text.UTF8Encoding $false))

Push-Location $RepoRoot
try {
    git add .debug/last_run.log | Out-Null
    git commit -m "debug: run $(Get-Date -Format 'yyyy-MM-dd HH:mm') exit=$exit" --allow-empty | Out-Null
    git push origin HEAD
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Pushed .debug/last_run.log (exit=$exit). Tell the Linux session to pull." -ForegroundColor Cyan
exit $exit
