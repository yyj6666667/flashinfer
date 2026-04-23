$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "Continue"

$sitepkg = "C:\flashinfer\.venv\Lib\site-packages\kt_kernel"
if (-not (Test-Path $sitepkg)) { Write-Output "kt_kernel not installed"; exit 1 }

Write-Output "--- kt_kernel dir contents ---"
Get-ChildItem $sitepkg -File | Select-Object Name, @{N="KB";E={[math]::Round($_.Length/1KB)}} | Format-Table -AutoSize | Out-String

# Copy hwloc runtime DLL next to .pyd
$src = "C:\hwloc\bin\libhwloc-15.dll"
$dst = Join-Path $sitepkg "libhwloc-15.dll"
if (Test-Path $src) {
    Copy-Item -Path $src -Destination $dst -Force
    Write-Output ("copied: " + $dst)
} else {
    Write-Output "ERROR: $src missing"
    exit 2
}

# Retry import
& "C:\flashinfer\.venv\Scripts\python.exe" -c "import kt_kernel; print('OK'); print('wrapper:', kt_kernel.KTMoEWrapper)"
