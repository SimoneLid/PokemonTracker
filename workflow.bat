@echo off
setlocal enabledelayedexpansion

REM --- 1. PULIZIA INIZIALE (Opzionale ma raccomandata) ---
REM Pulisce le cartelle dataset prima di iniziare per evitare di mischiare dati vecchi
echo Sto pulendo le cartelle dataset esistenti...
if exist "dataset\images" rmdir /s /q "dataset\images"
if exist "dataset\labels" rmdir /s /q "dataset\labels"

REM --- 2. CICLO SU TUTTI I VIDEO ---
echo Inizio elaborazione dei video nella cartella videos\train...
echo.

REM Cerca tutti i file .mp4 nella cartella specificata
for %%f in (videos\train\*.mp4) do (
    REM Ottieni il nome del file senza percorso ed estensione (es. "pikachu")
    set "filename=%%~nf"
    
    REM Definisci i percorsi completi per video e xml
    set "video_path=videos\train\%%~nf.mp4"
    set "xml_path=videos\train\%%~nf.xml"
    
    REM Controlla se esiste il file XML corrispondente
    if exist "!xml_path!" (
        echo [INFO] Trovata coppia: !filename!
        
        echo    - Estrazione frame...
        python extract_frames.py "!video_path!"
        
        echo    - Parsing XML...
        python parse_xml.py "!xml_path!"
        
        echo [OK] !filename! elaborato.
        echo ---------------------------------------------
    ) else (
        echo [ATTENZIONE] XML non trovato per !video_path!. Salto...
        echo ---------------------------------------------
    )
)

REM --- 3. SPLIT TRAIN/VAL ---
REM Eseguiamo lo split una volta sola alla fine, dopo aver generato tutti i dati
echo.
echo Tutte le coppie elaborate. Eseguo split Train/Val...
python split_train_val.py

if exist "dataset\images" rmdir /s /q "dataset\images"
if exist "dataset\labels" rmdir /s /q "dataset\labels"

echo.
echo Processo completato!
pause