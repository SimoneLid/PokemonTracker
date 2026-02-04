import os
import shutil
from pathlib import Path
from sys import argv

# --- CONFIGURAZIONE ---
DATASET_ROOT = argv[1] if len(argv) > 1 else "dataset_single_v3"
SUBSETS = ["train", "val"]
TYPES = ["images", "labels"]

def parse_filename(filename):
    """
    Estrae SIA il nome del video CHE il numero del frame.
    Format atteso: frame_000000_Zona_2_01.jpg
    Restituisce una tupla: (video_name, frame_number)
    """
    # Rimuove estensione
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    
    # Controllo formato minimo: frame_XXXXXX_NomeVideo
    if len(parts) < 3:
        return None, None

    try:
        # parts[0] = "frame"
        # parts[1] = "000000" -> Frame Number
        frame_num = int(parts[1])
        
        # Tutto quello che c'è dopo il frame number è il nome del video
        # Ricostruiamo il nome unendo i pezzi dall'indice 2 in poi
        video_name = "_".join(parts[2:])
        
        return video_name, frame_num
    except ValueError:
        return None, None

def process_folder_and_organize(current_dir):
    """
    Scansiona una cartella piatta (es. images), raggruppa i file per video,
    calcola le sequenze e sposta i file nelle sottocartelle finali.
    """
    # 1. Raggruppiamo i file per VIDEO in memoria
    # Struttura: { "Zona_2_01": [ (0, file_path), (1, file_path) ], "AltroVideo": ... }
    video_groups = {}
    files_found = 0
    
    for file_path in current_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            video_name, frame_num = parse_filename(file_path.name)
            
            if video_name and frame_num is not None:
                if video_name not in video_groups:
                    video_groups[video_name] = []
                video_groups[video_name].append((frame_num, file_path))
                files_found += 1
            else:
                print(f" [!] Formato ignoto, salto: {file_path.name}")

    if files_found == 0:
        return 0, 0

    # 2. Processiamo ogni video per trovare le sequenze e spostare
    moved_count = 0
    sequences_created = 0

    for video_name, frames_list in video_groups.items():
        # Ordina i frame numericamente (FONDAMENTALE)
        frames_list.sort(key=lambda x: x[0])

        # Logica delle sequenze
        current_seq = []
        last_frame = -999
        seq_idx = 0

        # Funzione helper per spostare una lista di frame
        def move_sequence(files_to_move, v_name, s_idx):
            if not files_to_move: return
            
            # Crea cartella: images/NomeVideo/seq_XX
            seq_folder_name = f"seq_{s_idx:02d}"
            target_dir = current_dir / v_name / seq_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for _, f_path in files_to_move:
                dest = target_dir / f_path.name
                shutil.move(str(f_path), str(dest))
            
            return 1 # Conta come 1 sequenza creata

        # Loop sui frame ordinati
        for frame_num, file_path in frames_list:
            if not current_seq or frame_num == last_frame + 1:
                current_seq.append((frame_num, file_path))
            else:
                # Sequenza interrotta, salva la precedente
                sequences_created += move_sequence(current_seq, video_name, seq_idx)
                moved_count += len(current_seq)
                
                # Resetta per nuova sequenza
                seq_idx += 1
                current_seq = [(frame_num, file_path)]
            
            last_frame = frame_num

        # Salva l'ultima sequenza rimasta
        if current_seq:
            sequences_created += move_sequence(current_seq, video_name, seq_idx)
            moved_count += len(current_seq)

    return moved_count, sequences_created

def organize_dataset_full():
    base_path = Path(DATASET_ROOT)

    if not base_path.exists():
        print(f"ERRORE: La cartella '{DATASET_ROOT}' non esiste!")
        return

    print(f"Inizio riorganizzazione completa di '{DATASET_ROOT}'...\n")

    total_files = 0
    total_seqs = 0

    for subset in SUBSETS:      # train, val
        for dtype in TYPES:     # images, labels
            
            # Percorso corrente: es. dataset_single_v3/train/images
            current_dir = base_path / subset / dtype
            
            if not current_dir.exists():
                print(f"Saltato (non esiste): {current_dir}")
                continue

            print(f"Analisi e organizzazione: {current_dir}...")
            
            # Lancia la funzione principale
            moved, seqs = process_folder_and_organize(current_dir)
            
            if moved > 0:
                print(f"   -> Spostati {moved} file in {seqs} sottocartelle sequenziali.")
                total_files += moved
                total_seqs += seqs

    print("\n" + "="*30)
    print("OPERAZIONE COMPLETATA")
    print(f"Totale file spostati:   {total_files}")
    print(f"Totale sequenze create: {total_seqs}")
    
    if total_seqs > 0:
        avg = total_files / total_seqs
        print(f"Media file per cartella: {avg:.2f}")
    else:
        print("Media file per cartella: 0")
    print("="*30)

if __name__ == "__main__":
    confirm = input(f"Stai per riorganizzare '{DATASET_ROOT}' (da FLAT a Video/Seq). Continuare? (s/n): ")
    if confirm.lower() == 's':
        organize_dataset_full()
    else:
        print("Annullato.")