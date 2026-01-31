import cv2
import os
import glob
import shutil
from sys import argv

# --- CONFIGURAZIONE ---
IMAGES_DIR = "images_all"  # Ho corretto lo slash per compatibilità cross-platform
EXIT_DIR = "dataset_micro"
VIDEO_NAME = argv[1]
VIDEO_DIM = argv[2]
# Definisci colori diversi per classi diverse (B, G, R)
# Classe 0: Verde, 1: Rosso, 2: Blu, 3: Giallo, 4: Ciano, 5: Magenta
CLASS_COLORS = [
    (0, 255, 0),    # 0
    (0, 0, 255),    # 1
    (255, 0, 0),    # 2
    (0, 255, 255),  # 3
    (255, 255, 0),  # 4
    (255, 0, 255)   # 5
]
DEFAULT_COLOR = (200, 200, 200) # Colore mentre disegni
THICKNESS = 2
# ----------------------

drawing = False
ix, iy = -1, -1

# Lista di tuple (x1, y1, x2, y2, class_id)
current_boxes = [] 

# Variabile per gestire il box appena disegnato in attesa di assegnazione classe
pending_box = None 

def convert_to_yolo(size, box):
    """Converte coordinate pixel (x1, y1, x2, y2) in formato YOLO (xc, yc, w, h)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def load_existing_labels(txt_path, img_width, img_height):
    """Carica le label esistenti preservando la classe"""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x_n, y_n, w_n, h_n = map(float, parts)
                    class_id = int(c) # Recupero la classe dal file
                    
                    w = w_n * img_width
                    h = h_n * img_height
                    x = x_n * img_width
                    y = y_n * img_height
                    
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    boxes.append((x1, y1, x2, y2, class_id))
    return boxes

def save_labels(img_path, txt_path, boxes, img_width, img_height):
    """Salva la lista di box nel file .txt usando la classe specifica di ogni box"""
    if len(boxes) == 0:
        print(f"Immagine skippata (0 box).")
        return

    # Crea cartelle se non esistono (safety check)
    if not os.path.exists(os.path.join(EXIT_DIR, "images")):
        os.makedirs(os.path.join(EXIT_DIR, "images"))
        
    shutil.copy(img_path, os.path.join(EXIT_DIR, "images"))

    with open(txt_path, 'w') as f:
        for box in boxes:
            # box è (x1, y1, x2, y2, class_id)
            x1, y1, x2, y2, class_id = box
            
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            
            yolo_box = convert_to_yolo((img_width, img_height), (xmin, ymin, xmax, ymax))
            # Scrivo class_id invece che CLASS_ID globale
            f.write(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
    print(f"Salvato: {os.path.basename(txt_path)} con {len(boxes)} box.")

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, pending_box, temp_img, img

    # Se c'è un box in attesa di classe, il mouse è disabilitato
    if pending_box is not None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            # Disegna i box già confermati con i loro colori
            draw_confirmed_boxes(temp_img)
            # Disegna quello che sto trascinando ora (grigio neutro)
            cv2.rectangle(temp_img, (ix, iy), (x, y), DEFAULT_COLOR, 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 5 and abs(y - iy) > 5: # Filtro micro-click
            # Invece di salvare subito, lo metto in "pending"
            # (x1, y1, x2, y2) senza classe per ora
            pending_box = (ix, iy, x, y)
        
        # Aggiorna visualizzazione
        temp_img = img.copy()
        draw_confirmed_boxes(temp_img)

def draw_confirmed_boxes(image):
    """Funzione helper per disegnare tutti i box salvati"""
    for b in current_boxes:
        x1, y1, x2, y2, cid = b
        # Se la classe è fuori range colori, usa bianco
        color = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (255, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, THICKNESS)
        # Scrivi ID classe piccolo vicino al box
        cv2.putText(image, str(cid), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    global img, temp_img, current_boxes, pending_box
    
    # Setup cartelle
    if not os.path.exists(EXIT_DIR):
        os.makedirs(os.path.join(EXIT_DIR, "labels"))
        os.makedirs(os.path.join(EXIT_DIR, "images"))
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    img_files.sort()
    
    if not img_files:
        print(f"Nessuna immagine trovata in {IMAGES_DIR}")
        return

    print("--- LABELER MULTI-CLASSE ---")
    print("ISTRUZIONI:")
    print(" 1. TRASCINA MOUSE: Disegna box")
    print(" 2. RILASCIA MOUSE: Il programma chiederà la classe")
    print(" 3. PREMI TASTO [0-5]: Assegna la classe al box")
    print("-" * 30)
    print(" [D]: Prossima immagine (Salva)")
    print(" [A]: Salta immagine")
    print(" [C]: Cancella tutto")
    print(" [Z]: Cancella ultimo box")
    print(" [Q]: Esci")

    cv2.namedWindow('Labeler')
    cv2.setMouseCallback('Labeler', mouse_callback)

    idx = 0
    saved_img = [0 for x in range(6)]
    
    while idx < len(img_files):
        img_path = img_files[idx]
        filename = os.path.basename(img_path)
        
        if VIDEO_NAME not in filename:
            idx += 1
            continue

        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(EXIT_DIR, "labels", txt_name)

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
            
        h, w = img.shape[:2]
        
        # Reset stato per nuova immagine
        current_boxes = load_existing_labels(txt_path, w, h)
        pending_box = None
        
        temp_img = img.copy()
        draw_confirmed_boxes(temp_img)

        while True:
            display = temp_img.copy()

            # --- LOGICA VISUALIZZAZIONE ---
            if pending_box is not None:
                # Se siamo in attesa di classe, disegna il box temporaneo in BIANCO SPESSO
                px1, py1, px2, py2 = pending_box
                cv2.rectangle(display, (px1, py1), (px2, py2), (255, 255, 255), 2)
                
                # Istruzione a schermo gigante
                cv2.putText(display, "PREMI 0-5 PER CLASSE", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Info immagine
            info_text = f"{idx+1}/{VIDEO_DIM}: {filename} | Box: {len(current_boxes)}"
            cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Labeler', display)
            key = cv2.waitKey(20) & 0xFF

            # --- GESTIONE INPUT ---
            
            # 1. Se stiamo aspettando la classe (Pending Mode)
            if pending_box is not None:
                # Controlla se è un numero tra 0 e 9
                if ord('0') <= key <= ord('9'):
                    class_id = key - ord('0') # Converte ASCII in intero
                    
                    # Salva il box con la classe scelta
                    px1, py1, px2, py2 = pending_box
                    current_boxes.append((px1, py1, px2, py2, class_id))
                    
                    pending_box = None # Reset pending
                    
                    # Aggiorna grafica
                    temp_img = img.copy()
                    draw_confirmed_boxes(temp_img)
                
                elif key == 27 or key == ord('x'): # ESC o x per annullare box
                    pending_box = None
                    temp_img = img.copy()
                    draw_confirmed_boxes(temp_img)
                
                # Se in pending, ignora altri tasti (d, a, q, etc.)
                continue 

            # 2. Comandi normali (se NON siamo in attesa di classe)
            if key == ord('d'): # Next & Save
                save_labels(img_path, txt_path, current_boxes, w, h)
                idx += 1
                for _,_,_,_,cid in current_boxes:
                    saved_img[cid]+=1
                print("Immagini attuali:")
                for x in range(6):
                    print(f"- Classe {x}: {saved_img[x]}")
                break
            elif key == ord('a'): # Skip
                idx += 1
                break
            elif key == ord('c'): # Clear all
                current_boxes = []
                temp_img = img.copy()
            elif key == ord('z'): # Undo Last
                if len(current_boxes) > 0:
                    current_boxes.pop()
                    temp_img = img.copy()
                    draw_confirmed_boxes(temp_img)
            elif key == ord('q'): # Quit
                save_labels(img_path, txt_path, current_boxes, w, h)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("---------------------------")
    for x in range(6):
        print(f"Salvate {saved_img[x]} immagini contenenti la classe {x}.")
    print("---------------------------")

if __name__ == "__main__":
    main()