@echo off
setlocal enabledelayedexpansion

set MIN=5
set MAX=9
set DATASET_PATH=dataset_stream/val


echo Scan from stream%MIN% to stream%MAX%
echo.

:: Cicle all the streams
for /L %%i in (%MIN%, 1, %MAX%) do (
    set "WEIGHTS_DIR=yolov5\runs\train\stream%%i\weights"
    
    echo CURRENT DIR: !WEIGHTS_DIR!

    if exist "!WEIGHTS_DIR!" (
        :: Cicle for all the weights
        for %%f in ("!WEIGHTS_DIR!\*.pt") do (
            set "WEIGHT_PATH=%%f"
            
            echo [TESTING] Model: %%~nxf
            
            python scripts/metrics.py --weights "!WEIGHT_PATH!" --data "%DATASET_PATH%"
        )
    ) else (
        echo [ERROR] Dir not found: !WEIGHTS_DIR!
    )
    echo.
)

echo.
echo All metrics created
pause