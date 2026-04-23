$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "Stop"

$zip = "C:\hwloc-win64.zip"
$target = "C:\hwloc"
if (Test-Path $target) { Remove-Item -Recurse -Force $target }

# Extract
Expand-Archive -Path $zip -DestinationPath "C:\hwloc_extract" -Force
# Archive contains hwloc-win64-build-2.11.2\ — flatten
$inner = Get-ChildItem "C:\hwloc_extract" -Directory | Select-Object -First 1
Move-Item $inner.FullName $target
Remove-Item -Recurse -Force "C:\hwloc_extract"

Write-Output "--- hwloc layout ---"
Get-ChildItem $target -Directory | Select-Object -ExpandProperty Name
Write-Output "--- include/hwloc.h ---"
Test-Path (Join-Path $target "include\hwloc.h")
Write-Output "--- lib ---"
Get-ChildItem (Join-Path $target "lib") -File | Select-Object -ExpandProperty Name
Write-Output "--- bin ---"
Get-ChildItem (Join-Path $target "bin") -File | Select-Object -ExpandProperty Name
