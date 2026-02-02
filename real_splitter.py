import os
import shutil
import random
from pathlib import Path

# --- CONFIGURAZIONE ---
SOURCE_DIR = "dataset_single_v3"  # La tua cartella attuale
DEST_DIR = "dataset_single_v3_final_split"  # La nuova cartella che verrà creata

# Imposta il seed per rendere lo split riproducibile (sempre uguale se rilanci)
random.seed(42)

# Definisci quali video vanno nel VAL (esclusi background e video 07)
# Esempio: Video 00, 01, 02, 03 -> TRAIN. Video 04, 05 -> VAL.
VAL_VIDEOS = ['03', '05'] 

def setup_directories():
    """Crea la struttura delle cartelle vuote"""
    for split in ['train', 'val']:
        for type_ in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, type_), exist_ok=True)

def is_background(label_path):
    """Controlla se un file label è vuoto o non esiste (background)"""
    if not os.path.exists(label_path):
        return True # Se manca il txt, è background
    
    # Controlla se il file è vuoto
    with open(label_path, 'r') as f:
        lines = f.readlines()
        # Rimuove righe vuote o solo spazi
        content = [line.strip() for line in lines if line.strip()]
        return len(content) == 0

def get_video_id(filename):
    """Estrae l'ID del video dal nome file.
    Esempio: frame_000260_Zona_2_01.jpg -> '01'
    """
    name_without_ext = os.path.splitext(filename)[0]
    # Splitta per '_' e prende l'ultima parte
    parts = name_without_ext.split('_')
    if parts:
        return parts[-1]
    return "unknown"

def process_dataset():
    setup_directories()
    
    # Raccogli tutti i file immagini (sia da train che da val attuali per ri-mischiare tutto)
    all_images = []
    # Cerca ricorsivamente in tutte le sottocartelle
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(list(Path(SOURCE_DIR).rglob(ext)))

    print(f"Trovati {len(all_images)} frame totali. Inizio riorganizzazione...")
    
    count_train = 0
    count_val = 0
    count_bg = 0

    for img_path in all_images:
        filename = img_path.name
        
        # Trova il file label corrispondente
        # Assume che il label sia nella cartella 'labels' parallela a 'images'
        # O nella stessa struttura. Cerchiamo di ricostruire il path.
        label_name = os.path.splitext(filename)[0] + ".txt"
        
        # Risaliamo alla cartella padre (es. 'train') e cerchiamo in 'labels'
        parent_dir = img_path.parent.parent # dataset/train
        label_path = parent_dir / "labels" / label_name
        
        # Se non lo trova lì, prova a cercarlo nello stesso folder (caso flat)
        if not label_path.exists():
            label_path = img_path.parent / label_name

        # --- LOGICA DI DIVISIONE ---
        destination_split = "train" # Default

        # 1. Analisi Background
        if is_background(label_path):
            destination_split = "train" # User request: All backgrounds to train
            count_bg += 1
        else:
            # 2. Analisi Video ID
            vid_id = get_video_id(filename)
            
            if vid_id == '07':
                # Caso speciale Patrat: 80% Train, 20% Val (Random)
                if random.random() < 0.20:
                    destination_split = "val"
                else:
                    destination_split = "train"
            
            elif vid_id in VAL_VIDEOS:
                # Video interi dedicati alla validazione
                destination_split = "val"
            
            else:
                # Tutti gli altri video (00, 01, 02, 03...) vanno nel train
                destination_split = "train"

        # --- COPIA DEI FILE ---
        # Copia immagine
        shutil.copy2(img_path, os.path.join(DEST_DIR, destination_split, "images", filename))
        
        # Copia label (se esiste ed è vuoto lo copiamo vuoto, se ha dati lo copiamo)
        # Se era background e il file non esisteva, ne creiamo uno vuoto per YOLO
        target_label_path = os.path.join(DEST_DIR, destination_split, "labels", label_name)
        
        if label_path.exists():
            shutil.copy2(label_path, target_label_path)
        else:
            # Crea file vuoto per background senza label
            with open(target_label_path, 'w') as f:
                pass

        if destination_split == "train":
            count_train += 1
        else:
            count_val += 1

    print("-" * 30)
    print("Riorganizzazione Completata!")
    print(f"Nuovo Dataset creato in: {DEST_DIR}")
    print(f"Immagini in TRAIN: {count_train} (inclusi {count_bg} background)")
    print(f"Immagini in VAL:   {count_val}")
    print("-" * 30)

if __name__ == "__main__":
    process_dataset()