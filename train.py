from multiprocessing import freeze_support
from ultralytics import YOLO
from sys import argv
import torch
import shutil
import os

if __name__ == "__main__":
    if torch.cuda.is_available():
        freeze_support()

        model_name = argv[3]

        model = YOLO(argv[1]) 

        results = model.train(
            data=argv[2],
            epochs=100,          # Un po' di più perché togliamo augmentation forte
            imgsz=640,
            batch=16,
            rect=True,
            # --- AUGMENTATION CORRETTA PER VIDEO GAME ---
            degrees=5.0,         # Riduci: i pokemon non ruotano di 45 gradi a caso
            translate=0.1,       # Riduci
            scale=0.5,           # Ok
            fliplr=0.5,          # Ok
            mosaic=1.0,          # Ok
            # --- COLORI (CRITICO) ---
            hsv_h=0.015,         # Tonalità: minima variazione
            hsv_s=0.2,           # Saturazione: bassa variazione (fondamentale per Magikarp)
            hsv_v=0.3,           # Valore: media variazione (luce/ombra)
            
            # --- TRAINING TRICKS ---
            close_mosaic=15,     # Disabilita mosaico alla fine per precisione
            warmup_epochs=5,     
        )
        print("\n--- Inizio salvataggio del modello migliore ---")
        source_path = model.trainer.best

        dest_folder = "models"
        os.makedirs(dest_folder, exist_ok=True)

        if not model_name.endswith('.pt'):
            model_name += '.pt'
            
        dest_path = os.path.join(dest_folder, model_name)

        try:
            shutil.copy(source_path, dest_path)
            print(f"Successo: Modello copiato da '{source_path}' a '{dest_path}'")
        except Exception as e:
            print(f"Errore durante la copia del modello: {e}")

    else:
        print("Cuda non presente sulla macchina")