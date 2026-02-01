@echo off
setlocal enabledelayedexpansion

set "num_done=3"
set "dataset_out=dataset_single_v2"
set "model_path=models/single_v1_micro_trained.pt"

REM --- CICLO SU TUTTI I VIDEO ---
echo Inizio elaborazione dei video nella cartella videos\train...

REM Cerca tutti i file .mp4 nella cartella specificata
for %%f in (videos\train\*.mp4) do (
    REM Ottieni il nome del file senza percorso ed estensione (es. "pikachu")
    set "filename=%%~nf"
    

    REM La sintassi ~-2 significa "parti dalla fine e prendi 2 caratteri"
    set "video_num=!filename:~-2!"

    echo Controllo il numero: !video_num!.

    REM Confronto numerico (GTR sta per "Greater Than")
    if !video_num! GTR %num_done% (
        echo Accettato: !video_num! e maggiore di %num_done%.

        REM Definisci i percorsi completi per video
        set "video_path=videos\train\%%~nf.mp4"
        
    
        echo    - Calcolo frame...
        py selector.py "models/single_v1_micro_trained.pt" "!video_path!" %dataset_out%

        
        echo [OK] !filename! elaborato.
        echo ---------------------------------------------
    )

    

)




REM --- SPLIT TRAIN/VAL ---
REM Eseguiamo lo split una volta sola alla fine, dopo aver generato tutti i dati
echo.
echo Tutte le coppie elaborate. Eseguo split Train/Val...
python split_train_val.py %dataset_out% %dataset_out%


if exist "%dataset_out%\images" rmdir /s /q "%dataset_out%\images"
if exist "%dataset_out%\labels" rmdir /s /q "%dataset_out%\labels"

echo.
echo Processo completato!
pause