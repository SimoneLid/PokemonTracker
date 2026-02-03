import sys
import os
from pathlib import Path
import time
import torch
import cv2
import numpy as np

# --- 1. CONFIGURAZIONE PATH E IMPORT ---
# Ottieni il percorso assoluto della cartella corrente
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Cartella ComputerVision

# Aggiungi la cartella 'yolov5' al system path
YOLO_PATH = ROOT / "yolov5"
if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

# Import specifici di YOLOv5
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

# --- 2. CARICAMENTO MODELLO ---
print("Caricamento modello...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorso pesi (adattato per puntare dentro la cartella yolov5 se necessario)
# Se runs è dentro yolov5, usiamo YOLO_PATH / ...
weights_path = YOLO_PATH / "runs" / "train" / "exp10" / "weights" / "best.pt"

# Caricamento con torch.hub locale
# path=str(YOLO_PATH) dice a torch di cercare il codice sorgente nella cartella yolov5
model = torch.hub.load(str(YOLO_PATH), 'custom', path=str(weights_path), source='local')
model.to(device).eval()

print(f"Modello caricato su {device}")

# --- 3. CONFIGURAZIONE VIDEO ---
video_path = r"C:\Users\pietr\Desktop\ComputerVision\ComputerVision\videos\test\Zona_2_09.mp4"
cap = cv2.VideoCapture(video_path)
WINDOW_NAME = "Supervisione 6-Canali"

# Variabili di stato
prev_frame = None  # Buffer per il frame t-1
img_size = 640     # Dimensione input (deve combaciare col training)
frame_count = 0

print("Inizio inferenza. Premi 'q' per uscire.")

# --- 4. LOOP PRINCIPALE ---
while cap.isOpened():
    # Timer start
    t0 = time.time()
    
    success, frame = cap.read()
    if not success:
        print("Fine video o errore lettura.")
        break
        
    frame_count += 1
    frame_orig = frame.copy()

    # --- A. PRE-PROCESSING (Letterbox + Stacking) ---
    # Ridimensiona mantenendo aspect ratio
    img_resized = letterbox(frame, img_size, stride=32, auto=False)[0]

    # Gestione primo frame (se non c'è storico, duplica il frame attuale)
    if prev_frame is None:
        prev_frame = img_resized

    # Creazione stack 6 canali: (t-1) + (t)
    img_stacked = np.concatenate((prev_frame, img_resized), axis=2)
    
    # HWC to CHW, BGR to RGB
    img_tensor = img_stacked.transpose((2, 0, 1))[::-1] 
    img_tensor = np.ascontiguousarray(img_tensor)
    
    # Da Numpy a Torch Tensor
    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalizza 0-1
    
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]  # Aggiungi batch dimension (1, 6, 640, 640)
    
    t1 = time.time() # Fine Pre-proc

    # --- B. INFERENZA ---
    with torch.no_grad():
        pred = model(img_tensor)
    
    t2 = time.time() # Fine Inference

    # --- C. POST-PROCESSING (NMS) ---
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    
    t3 = time.time() # Fine NMS

    # --- D. DISEGNO E LOGGING ---
    det = pred[0]
    
    # Costruzione stringa di log
    s = f"Frame {frame_count}: "
    
    annotator = Annotator(frame_orig, line_width=2, example=str(model.names))

    if len(det):
        # Riscala le box coordinate modello -> coordinate video originale
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame_orig.shape).round()

        # Conta classi
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            s += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "

        # Disegna box
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{model.names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            
        final_frame = annotator.result()
    else:
        s += "(no detections)"
        final_frame = frame_orig

    # Calcolo tempi
    dt_inf = (t2 - t1) * 1000
    dt_nms = (t3 - t2) * 1000
    
    # Stampa log
    print(f"{s} Done. ({dt_inf:.1f}ms inf, {dt_nms:.1f}ms NMS)")

    # Aggiorna buffer per il prossimo giro
    prev_frame = img_resized

    # Mostra video
    cv2.imshow(WINDOW_NAME, final_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()