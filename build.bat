@echo off
set build_type="%~1"
if %build_type% == "install-dev" (
    py -m pip install --no-build-isolation --editable .
) else if %build_type% == "install" (
    py -m pip install
) else echo build.bat <build-type>: build-type must be dev or install

