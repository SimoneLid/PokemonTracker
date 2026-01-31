@echo off
setlocal enabledelayedexpansion

REM --- 1. PULIZIA INIZIALE (Opzionale ma raccomandata) ---
REM Pulisce le cartelle dataset prima di iniziare per evitare di mischiare dati vecchi
echo Sto pulendo le cartelle dataset esistenti...
if exist "dataset_single" rmdir /s /q "dataset_single"

REM --- 2. CICLO SU TUTTI I VIDEO ---
echo Inizio elaborazione dei video nella cartella videos\train...

REM Cerca tutti i file .mp4 nella cartella specificata
for %%f in (videos\train\*.mp4) do (
    REM Ottieni il nome del file senza percorso ed estensione (es. "pikachu")
    set "filename=%%~nf"
    
    REM Definisci i percorsi completi per video
    set "video_path=videos\train\%%~nf.mp4"
    
   
    echo    - Calcolo frame...
    python selector.py "micro_model/micro.pt" "!video_path!" "dataset_single" "dataset_single"

    
    echo [OK] !filename! elaborato.
    echo ---------------------------------------------

)




REM --- 4. SPLIT TRAIN/VAL ---
REM Eseguiamo lo split una volta sola alla fine, dopo aver generato tutti i dati
echo.
echo Tutte le coppie elaborate. Eseguo split Train/Val...
python split_train_val.py "dataset_single" "dataset_single"


if exist "dataset_single\images" rmdir /s /q "dataset_single\images"
if exist "dataset_single\labels" rmdir /s /q "dataset_single\labels"

echo.
echo Processo completato!
pause