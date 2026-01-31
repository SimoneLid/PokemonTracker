import os
import cv2
import re
import glob
from sys import argv

# --- CONFIGURAZIONE ---
DATASET_ROOT = argv[1]  # La cartella che contiene train e val
VIDEOS_DIR = os.path.join("videos", "train") # Dove si trovano i video sorgente
EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'] # Estensioni video supportate

# Regex per parsare il nome file: frame_000010_Zona_2_02.txt
# Spiegazione: Cerchiamo "frame_", poi cifre (\d+), poi un underscore, 
# poi tutto il resto (.+) che è il nome video, fino a .txt
FILENAME_REGEX = re.compile(r"frame_(\d+)_(.+)\.txt")

def get_video_map(video_folder):
    """
    Crea una mappa {nome_video_senza_ext: percorso_completo}
    Esempio: {'Zona_2_02': 'videos/train/Zona_2_02.mp4'}
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
    Ritorna: dict { 'NomeVideo': [ (frame_idx, percorso_output_img), ... ] }
    """
    tasks = {}
    
    # Scansioniamo sia 'train' che 'val'
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_root, split, "labels")
        images_dir = os.path.join(dataset_root, split, "images")
        
        # Crea cartella immagini se non esiste
        os.makedirs(images_dir, exist_ok=True)
        
        if not os.path.exists(labels_dir):
            continue

        print(f"Scansione label in: {labels_dir}...")
        
        txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            
            # Parsing del nome file
            match = FILENAME_REGEX.match(filename)
            if match:
                frame_num_str = match.group(1) # Es: "000010"
                video_name = match.group(2)    # Es: "Zona_2_02"
                frame_idx = int(frame_num_str)
                
                # Definiamo dove salvare l'immagine (stesso nome del txt ma .jpg)
                img_name = filename.replace(".txt", ".jpg")
                dest_path = os.path.join(images_dir, img_name)
                
                # Aggiungiamo al task list
                if video_name not in tasks:
                    tasks[video_name] = []
                
                tasks[video_name].append({
                    'frame_idx': frame_idx,
                    'dest_path': dest_path,
                    'txt_source': filename # Solo per log
                })
            else:
                print(f"[WARN] Nome file non valido ignorato: {filename}")

    return tasks

def extract_frames(tasks, video_map):
    print("\n--- Inizio Estrazione Frame ---")
    
    total_extracted = 0
    total_skipped = 0
    
    for video_name, requests in tasks.items():
        # Controlliamo se abbiamo il video
        if video_name not in video_map:
            print(f"[ERRORE] Video non trovato per i label: {video_name}. Saltati {len(requests)} frame.")
            continue
            
        video_path = video_map[video_name]
        
        # Filtriamo le richieste: processiamo solo se l'immagine NON esiste già
        pending_requests = [r for r in requests if not os.path.exists(r['dest_path'])]
        skipped_requests = len(requests) - len(pending_requests)
        total_skipped += skipped_requests
        
        if not pending_requests:
            print(f"[SKIP] {video_name}: Tutti i {len(requests)} frame esistono già.")
            continue

        print(f"[PROCESS] {video_name}: Estraggo {len(pending_requests)} frame (skippati {skipped_requests})...")
        
        # Ordiniamo le richieste per numero frame (importante per seek efficiente)
        pending_requests.sort(key=lambda x: x['frame_idx'])
        
        # Apriamo il video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERRORE] Impossibile aprire il video: {video_path}")
            continue
            
        for req in pending_requests:
            frame_idx = req['frame_idx']
            dest_path = req['dest_path']
            
            # Imposta la posizione del video al frame specifico
            # Nota: CAP_PROP_POS_FRAMES è 0-based.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(dest_path, frame)
                total_extracted += 1
            else:
                print(f"[ERRORE] Impossibile leggere frame {frame_idx} da {video_name} (Video finito?)")
        
        cap.release()

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