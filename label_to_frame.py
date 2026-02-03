import os
import cv2
import re
import glob
from sys import argv

# --- CONFIGURAZIONE ---
# Assicurati di passare il percorso dataset come primo argomento
if len(argv) < 2:
    print("ERRORE: Specifica il percorso del dataset come argomento.")
    exit()

DATASET_ROOT = argv[1]  # La cartella che contiene train e val
VIDEOS_DIR = os.path.join("videos", "used") # Dove si trovano i video sorgente
EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'] # Estensioni video supportate

# Regex per parsare il nome file: frame_000010_Zona_2_02.txt
FILENAME_REGEX = re.compile(r"frame_(\d+)_(.+)\.txt")

def get_video_map(video_folder):
    """
    Crea una mappa {nome_video_senza_ext: percorso_completo}
    """
    video_map = {}
    if not os.path.exists(video_folder):
        print(f"[ERRORE] La cartella video non esiste: {video_folder}")
        return video_map

    for f in os.listdir(video_folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in EXTENSIONS:
            video_map[name] = os.path.join(video_folder, f)
    
    print(f"--- Trovati {len(video_map)} video sorgente in {video_folder} ---")
    return video_map

def scan_labels_and_build_tasks(dataset_root):
    """
    Scansiona train e val e raggruppa le richieste per video.
    """
    tasks = {}
    
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_root, split, "labels")
        images_dir = os.path.join(dataset_root, split, "images")
        
        os.makedirs(images_dir, exist_ok=True)
        
        if not os.path.exists(labels_dir):
            continue

        print(f"Scansione label in: {labels_dir}...")
        
        txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            
            match = FILENAME_REGEX.match(filename)
            if match:
                frame_num_str = match.group(1)
                video_name = match.group(2)
                frame_idx = int(frame_num_str)
                
                img_name = filename.replace(".txt", ".jpg")
                dest_path = os.path.join(images_dir, img_name)
                
                if video_name not in tasks:
                    tasks[video_name] = []
                
                tasks[video_name].append({
                    'frame_idx': frame_idx,
                    'dest_path': dest_path
                })
            else:
                # print(f"[WARN] Nome file non valido ignorato: {filename}")
                pass

    return tasks

def extract_frames(tasks, video_map):
    print("\n--- Inizio Estrazione Frame (Modalità Sequenziale) ---")
    
    total_extracted = 0
    total_skipped = 0
    
    for video_name, requests in tasks.items():
        # 1. Check video
        if video_name not in video_map:
            print(f"[ERRORE] Video non trovato: {video_name}. Saltati {len(requests)} frame.")
            continue
            
        video_path = video_map[video_name]
        
        # 2. Filtra richieste già soddisfatte
        pending_requests = [r for r in requests if not os.path.exists(r['dest_path'])]
        skipped = len(requests) - len(pending_requests)
        total_skipped += skipped
        
        if not pending_requests:
            print(f"[SKIP] {video_name}: Completato (tutti i file esistono).")
            continue

        # 3. Prepara lookup veloce
        # Creiamo un dizionario { indice_frame: percorso_destinazione }
        # Se ci sono più richieste per lo stesso frame (raro), sovrascrive, ma va bene
        targets = { req['frame_idx']: req['dest_path'] for req in pending_requests }
        
        # Troviamo il frame massimo necessario per fermare la lettura del video appena finito
        max_frame_needed = max(targets.keys())
        
        print(f"[PROCESS] {video_name}: Estraggo {len(targets)} frame (stop al frame {max_frame_needed})...")
        
        # 4. Scansione Video Sequenziale
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERRORE] Impossibile aprire: {video_path}")
            continue

        current_frame_idx = 0
        extracted_count = 0
        
        while True:
            # Ottimizzazione: Se abbiamo superato l'ultimo frame che ci serve, usciamo
            if current_frame_idx > max_frame_needed:
                break
            
            ret, frame = cap.read()
            if not ret:
                break # Fine del video naturale
            
            # Controlliamo se questo indice è nella lista dei desideri (Lookup O(1))
            if current_frame_idx in targets:
                dest_path = targets[current_frame_idx]
                cv2.imwrite(dest_path, frame)
                extracted_count += 1
                
                # Feedback visivo ogni tanto
                if extracted_count % 10 == 0:
                    print(f"  -> Estratto frame {current_frame_idx}...", end='\r')

            current_frame_idx += 1
        
        cap.release()
        total_extracted += extracted_count
        print(f"  -> {video_name}: Finito. {extracted_count} nuove immagini create.\n")

    print("\n--- RIEPILOGO ---")
    print(f"Frame estratti e salvati: {total_extracted}")
    print(f"Frame già esistenti (skippati): {total_skipped}")

if __name__ == "__main__":
    # 1. Trova i percorsi dei video reali
    video_mapping = get_video_map(VIDEOS_DIR)
    
    if not video_mapping:
        print("Nessun video trovato. Controlla il percorso VIDEOS_DIR.")
        exit()

    # 2. Crea la lista di cose da fare leggendo i txt
    tasks_dict = scan_labels_and_build_tasks(DATASET_ROOT)
    
    if not tasks_dict:
        print("Nessun file label trovato o pattern non riconosciuto.")
        exit()
        
    # 3. Esegui l'estrazione
    extract_frames(tasks_dict, video_mapping)