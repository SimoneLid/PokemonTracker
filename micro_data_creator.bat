@echo off
setlocal enabledelayedexpansion

REM --- 1. PULIZIA INIZIALE (Opzionale ma raccomandata) ---
REM Pulisce le cartelle dataset prima di iniziare per evitare di mischiare dati vecchi
echo Sto pulendo le cartelle dataset esistenti...
if exist "images_all" rmdir /s /q "images_all"

REM --- 2. CICLO SU TUTTI I VIDEO ---
echo Inizio elaborazione dei video nella cartella videos\train...

REM Cerca tutti i file .mp4 nella cartella specificata
for %%f in (videos\train\*.mp4) do (
    REM Ottieni il nome del file senza percorso ed estensione (es. "pikachu")
    set "filename=%%~nf"
    
    REM Definisci i percorsi completi per video
    set "video_path=videos\train\%%~nf.mp4"
    
   
    echo    - Estrazione frame...
    python extract_frames.py "!video_path!" "images_all"

    
    echo [OK] !filename! elaborato.
    echo ---------------------------------------------
    set /p VALORE_ESTRATTO=<temp_data.txt

    REM --- 3. LABELING DEI PRIMI BOX ---
    py box_drawer.py "%%~nf" !VALORE_ESTRATTO!

)




REM --- 4. SPLIT TRAIN/VAL ---
REM Eseguiamo lo split una volta sola alla fine, dopo aver generato tutti i dati
echo.
echo Tutte le coppie elaborate. Eseguo split Train/Val...
python split_train_val.py "dataset_micro" "dataset_micro"


if exist "dataset_micro\images" rmdir /s /q "dataset_micro\images"
if exist "dataset_micro\labels" rmdir /s /q "dataset_micro\labels"

echo.
echo Processo completato!
pause