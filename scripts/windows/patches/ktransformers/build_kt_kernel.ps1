$ProgressPreference = "SilentlyContinue"
$ErrorActionPreference = "Continue"

# vcvars env (cl.exe + nvcc on PATH)
$envDump = "C:\flashinfer\.sglang_launch_logs\vcvars_env.txt"
Get-Content $envDump | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2] }
}
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

# kt-kernel build env
$env:CPUINFER_CPU_INSTRUCT = "AVX2"
$env:CPUINFER_USE_CUDA = "1"
# Extra CMake args:
#  - HWLOC_INCLUDE_DIR / HWLOC_LIBRARY from prebuilt open-mpi hwloc-2.11.2 at
#    C:\hwloc (vcpkg path gave up: it needed MSYS2 autotools to build hwloc
#    from source and that download chain timed out from fi-win).
#  - SM120 for RTX 5060 consumer Blackwell (default CMake list was 80;86;89;90)
$env:CMAKE_ARGS = "-DHWLOC_INCLUDE_DIR=C:/hwloc/include -DHWLOC_LIBRARY=C:/hwloc/lib/libhwloc.lib -DCMAKE_CUDA_ARCHITECTURES=120-real"

$py  = "C:\flashinfer\.venv\Scripts\python.exe"
$pip = "C:\flashinfer\.venv\Scripts\pip.exe"

Write-Output "=== env check ==="
Write-Output ("cl.exe   : " + (Get-Command cl.exe -ErrorAction SilentlyContinue).Source)
Write-Output ("nvcc.exe : " + (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source)
Write-Output ("cmake    : " + (Get-Command cmake -ErrorAction SilentlyContinue).Source)
Write-Output ("python   : $py")
Write-Output ("CPUINFER_CPU_INSTRUCT=" + $env:CPUINFER_CPU_INSTRUCT)
Write-Output ("CPUINFER_USE_CUDA=" + $env:CPUINFER_USE_CUDA)
Write-Output ("CMAKE_ARGS=" + $env:CMAKE_ARGS)

# Downgrade setuptools first to satisfy torch 2.11 constraint
& $pip install 'setuptools<82' 2>&1 | Select-Object -Last 3

Write-Output "`n=== pip install kt-kernel (no build isolation) ==="
Push-Location "C:\ktransformers\kt-kernel"
& $pip install . --no-build-isolation --verbose 2>&1
$rc = $LASTEXITCODE
Pop-Location

Write-Output "`n=== result ==="
if ($rc -eq 0) {
    Write-Output "pip install OK"
    & $py -c "import kt_kernel; print('kt_kernel module at:', kt_kernel.__file__); print('has KTMoEWrapper:', hasattr(kt_kernel, 'KTMoEWrapper'))"
} else {
    Write-Output ("pip install FAILED rc=" + $rc)
}
