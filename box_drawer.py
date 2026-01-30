import cv2
import os
import glob

# --- CONFIGURAZIONE ---
IMAGES_DIR = "dataset_micro\images"      # Cartella con le tue immagini
LABELS_DIR = "dataset_micro\labels"    # Cartella dove salvare i txt
CLASS_ID = 0               # 0 = Pikachu (o la tua classe)
# ----------------------

# Colori per il disegno (B, G, R)
COLOR_BOX = (0, 255, 0)    # Verde
THICKNESS = 2

drawing = False
ix, iy = -1, -1
current_boxes = [] # Lista di tuple (x1, y1, x2, y2) per l'immagine corrente

def convert_to_yolo(size, box):
    """Converte coordinate pixel (x1, y1, x2, y2) in formato YOLO (xc, yc, w, h)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # Normalizza
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def load_existing_labels(txt_path, img_width, img_height):
    """Carica le label esistenti se torni indietro su un'immagine già fatta"""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Riconverto da YOLO a Pixel per disegnarli
                    c, x_n, y_n, w_n, h_n = map(float, parts)
                    w = w_n * img_width
                    h = h_n * img_height
                    x = x_n * img_width
                    y = y_n * img_height
                    
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    boxes.append((x1, y1, x2, y2))
    return boxes

def save_labels(txt_path, boxes, img_width, img_height):
    """Salva la lista di box nel file .txt"""
    with open(txt_path, 'w') as f:
        for box in boxes:
            # Assicuriamoci che x1 < x2 e y1 < y2
            x1, y1, x2, y2 = box
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            
            yolo_box = convert_to_yolo((img_width, img_height), (xmin, ymin, xmax, ymax))
            f.write(f"{CLASS_ID} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
    print(f"Salvato: {txt_path} con {len(boxes)} box.")

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_boxes, temp_img, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Copia pulita dell'immagine per non lasciare scie
            temp_img = img.copy()
            # Disegna i box già confermati
            for b in current_boxes:
                cv2.rectangle(temp_img, (b[0], b[1]), (b[2], b[3]), COLOR_BOX, THICKNESS)
            # Disegna quello che sto trascinando ora
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 0, 255), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Aggiungi il box finale alla lista
        # Evita box di dimensione 0 (click accidentali)
        if abs(x - ix) > 2 and abs(y - iy) > 2:
            current_boxes.append((ix, iy, x, y))
        
        # Aggiorna l'immagine base
        temp_img = img.copy()
        for b in current_boxes:
            cv2.rectangle(temp_img, (b[0], b[1]), (b[2], b[3]), COLOR_BOX, THICKNESS)

def main():
    global img, temp_img, current_boxes
    
    # Setup cartelle
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)
        
    # Trova immagini
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    img_files.sort() # Ordina per nome
    
    if not img_files:
        print(f"Nessuna immagine trovata in {IMAGES_DIR}")
        return

    print("--- MANUAL LABELER ---")
    print("ISTRUZIONI:")
    print("  [MOUSE SX + TRASCINA]: Disegna box")
    print("  [D]: Prossima immagine (Salva)")
    print("  [A]: Immagine precedente (Salva)")
    print("  [C]: Cancella tutti i box in questa foto")
    print("  [Q]: Esci")
    print(f"Trovate {len(img_files)} immagini.")

    cv2.namedWindow('Labeler')
    cv2.setMouseCallback('Labeler', mouse_callback)

    idx = 0
    while idx < len(img_files):
        img_path = img_files[idx]
        filename = os.path.basename(img_path)
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(LABELS_DIR, txt_name)
        
        # Carica immagine
        img = cv2.imread(img_path)
        if img is None:
            print(f"Errore caricamento: {img_path}")
            idx += 1
            continue
            
        h, w = img.shape[:2]
        
        # Carica label esistenti se ci sono
        current_boxes = load_existing_labels(txt_path, w, h)
        
        # Prepara immagine temporanea per il display
        temp_img = img.copy()
        # Disegna i box esistenti
        for b in current_boxes:
            cv2.rectangle(temp_img, (b[0], b[1]), (b[2], b[3]), COLOR_BOX, THICKNESS)

        while True:
            # Mostra info a schermo
            display = temp_img.copy()
            cv2.putText(display, f"{idx+1}/{len(img_files)}: {filename}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(display, f"{idx+1}/{len(img_files)}: {filename}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Labeler', display)
            
            key = cv2.waitKey(20) & 0xFF

            # Navigazione
            if key == ord('d'): # Next
                save_labels(txt_path, current_boxes, w, h)
                idx += 1
                break
            elif key == ord('a'): # Prev
                save_labels(txt_path, current_boxes, w, h)
                idx = max(0, idx - 1)
                break
            elif key == ord('c'): # Clear
                current_boxes = []
                temp_img = img.copy()
            elif key == ord('q'): # Quit
                save_labels(txt_path, current_boxes, w, h)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Finito!")

if __name__ == "__main__":
    main()