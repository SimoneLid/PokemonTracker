@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURAZIONE ---
set MIN=5
set MAX=9
set DATA_PATH=dataset_stream/val

echo Inizio scansione profonda modelli da stream%MIN% a stream%MAX%...
echo.

:: 1. Ciclo sugli stream (es. stream1, stream2...)
for /L %%i in (%MIN%, 1, %MAX%) do (
    set "CURRENT_WEIGHTS_DIR=yolov5\runs\train\stream%%i\weights"
    
    echo =======================================================
    echo ANALISI CARTELLA: !CURRENT_WEIGHTS_DIR!
    echo =======================================================

    if exist "!CURRENT_WEIGHTS_DIR!" (
        :: 2. Ciclo su tutti i file .pt nella cartella weights
        for %%f in ("!CURRENT_WEIGHTS_DIR!\*.pt") do (
            set "FULL_PATH=%%f"
            echo.
            echo [TESTING] Modello: %%~nxf
            
            :: Esecuzione script Python
            python scripts/metrics.py --weights "!FULL_PATH!" --data "%DATA_PATH%"
        )
    ) else (
        echo [ERRORE] Cartella non trovata: !CURRENT_WEIGHTS_DIR!
    )
    echo.
)

echo.
echo Tutte le valutazioni sono state completate.
pause