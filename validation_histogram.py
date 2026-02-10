import os
import glob
import re
import sys
from pathlib import Path

# --- FUNZIONE PER TROVARE L'ULTIMO EXP ---
def get_latest_run_number(base_path):
    """
    Cerca nella cartella base tutte le sottocartelle che iniziano con 'exp'.
    Restituisce il numero più alto trovato.
    """
    path = Path(base_path)
    if not path.exists():
        raise FileNotFoundError(f"La cartella {base_path} non esiste.")
    
    exp_folders = [f for f in path.iterdir() if f.is_dir() and f.name.startswith('exp')]
    
    if not exp_folders:
        raise FileNotFoundError(f"Nessuna cartella 'exp' trovata in {base_path}")

    max_num = 0
    for folder in exp_folders:
        match = re.match(r'exp(\d*)', folder.name)
        if match:
            num_str = match.group(1)
            num = int(num_str) if num_str else 1 # 'exp' vale 1
            if num > max_num:
                max_num = num
    return max_num

# --- CONFIGURAZIONE ---
train_runs_path = "yolov5/runs/train" # Percorso base dove YOLO salva i training
data_yaml = "models/stream_single.yaml"      # Il tuo file .yaml del dataset
img_size = 640
conf_thres = 0.25
specific_weight_file = None           # Variabile per il caso di un singolo file

try:
    # --- GESTIONE ARGOMENTI (ARGV) ---
    if len(sys.argv) > 1:
        # CASO A: L'utente ha passato il numero della run (es: python script.py 78)
        run_number = int(sys.argv[1])
        exp_folder_name = "exp" if run_number == 1 else f"exp{run_number}"
        print(f"--> Modalità Manuale: Utilizzo run specificata {exp_folder_name}")
        
        # CASO B: L'utente ha passato anche il nome del peso (es: python script.py 78 best.pt)
        if len(sys.argv) > 2:
            specific_weight_file = sys.argv[2]
            print(f"--> Modalità Singolo File: Cerco solo '{specific_weight_file}'")

    else:
        # CASO C: Nessun argomento, comportamento automatico (es: python script.py)
        run_number = get_latest_run_number(train_runs_path)
        exp_folder_name = "exp" if run_number == 1 else f"exp{run_number}"
        print(f"--> Modalità Automatica: Rilevato ultimo esperimento {exp_folder_name}")

    # Costruisci i percorsi
    weights_folder = Path(train_runs_path) / exp_folder_name / "weights"
    project_dir_base = Path(train_runs_path) / exp_folder_name / "val_history"

    # Controlla se la cartella weights esiste
    if not weights_folder.exists():
        raise FileNotFoundError(f"La cartella weights non esiste qui: {weights_folder}")

except ValueError:
    print("ERRORE: Il primo argomento deve essere un numero intero (il numero della run).")
    exit()
except Exception as e:
    print(f"ERRORE: {e}")
    exit()

# ----------------------

# 3. Selezione dei file da processare
files = []

if specific_weight_file:
    # Se abbiamo specificato un file, cerchiamo solo quello
    target_file = weights_folder / specific_weight_file
    if target_file.exists():
        files.append(target_file)
    else:
        print(f"ERRORE: Il file '{specific_weight_file}' non esiste in {weights_folder}")
        exit()
else:
    # Altrimenti cerchiamo tutti gli epoch*.pt
    files = list(weights_folder.glob("epoch*.pt"))
    
    # Funzione per ordinare i file in base al numero (epoch10, epoch20...)
    def get_epoch_number(filepath):
        match = re.search(r'epoch(\d+)\.pt', filepath.name)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_epoch_number)

print(f"Trovati {len(files)} file da analizzare in {weights_folder}...")

# 4. Ciclo su ogni file
for f in files:
    # Determina il nome per la cartella di output
    if specific_weight_file:
        # Se è un file specifico (es. best.pt), usiamo il nome del file senza estensione
        run_name = f.stem 
        print(f"\n--- Analisi File {f.name} ---")
    else:
        # Se è una epoch automatica, usiamo il numero
        epoch_num = get_epoch_number(f)
        run_name = f"epoch_{epoch_num}"
        print(f"\n--- Analisi Epoca {epoch_num} ---")
    
    # Costruisci ed esegui il comando
    cmd = (
        f"py -3.10 yolov5/val.py "
        f"--weights {f} "
        f"--data {data_yaml} "
        f"--img {img_size} "
        f"--conf {conf_thres} "
        f"--project {project_dir_base} "
        f"--name {run_name} "
        f"--exist-ok"
    )
    
    os.system(cmd)

print(f"\nFinito! Trovi tutti i grafici nella cartella: {project_dir_base}")