@echo off
setlocal
cd /d "%~dp0\.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set TCNN_CUDA_ARCHITECTURES=86
set MAX_JOBS=4
set DISTUTILS_USE_SDK=1
set MSSdk=1
set PATH=%CD%\.venv-gpu\Scripts;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64;%PATH%
.venv-gpu\Scripts\python.exe -m pip install --no-build-isolation --no-deps third_party/tiny-cuda-nn/bindings/torch --disable-pip-version-check
exit /b %errorlevel%
