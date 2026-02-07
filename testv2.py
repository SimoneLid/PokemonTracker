import sys
import os
from pathlib import Path
import time
import torch
import cv2
import numpy as np
from sys import argv

# --- 1. CONFIGURAZIONE PATH E IMPORT ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
YOLO_PATH = ROOT / "yolov5"
if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

from utils.general import non_max_suppression, scale_boxes 
from utils.plots import Annotator, colors 
from utils.augmentations import letterbox 

# --- 2. CARICAMENTO MODELLO ---
print("Caricamento modello...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assicurati che il percorso pesi sia corretto
weights_path = YOLO_PATH / "runs" / "train" / "pippa" / "weights" / "best.pt"

# Caricamento modello
model = torch.hub.load(str(YOLO_PATH), 'custom', path=str(weights_path), source='local')
model.to(device).eval()

print(f"Modello caricato su {device}")

# --- 3. CONFIGURAZIONE VIDEO ---
# Gestione argomento da riga di comando o default
video_path = argv[1]
cap = cv2.VideoCapture(video_path)
WINDOW_NAME = "Supervisione 3-Canali (Standard)"

img_size = 640  # Dimensione input
frame_count = 0

print("Inizio inferenza. Premi 'q' per uscire.")

# --- 4. LOOP PRINCIPALE ---
while cap.isOpened():
    t0 = time.time()
    
    success, frame = cap.read()
    if not success:
        print("Fine video o errore lettura.")
        break
        
    frame_count += 1
    frame_orig = frame.copy()

    # --- A. PRE-PROCESSING (Adattato al tuo __getitem__) ---
    # 1. Letterbox (ridimensionamento con padding)
    img_resized = letterbox(frame, img_size, stride=32, auto=False)[0]

    # 2. Trasformazioni colori e assi (Come nel tuo __getitem__)
    # HWC to CHW, BGR to RGB
    img_tensor = img_resized.transpose((2, 0, 1))[::-1] 
    img_tensor = np.ascontiguousarray(img_tensor)
    
    # 3. Da Numpy a Torch Tensor
    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalizza 0-1
    
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]  # Aggiungi batch dimension (1, 3, 640, 640)
    
    t1 = time.time()

    # --- B. INFERENZA ---
    with torch.no_grad():
        pred = model(img_tensor)
    
    t2 = time.time()

    # --- C. POST-PROCESSING (NMS) ---
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    
    t3 = time.time()

    # --- D. DISEGNO E LOGGING ---
    det = pred[0]
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
    
    print(f"{s} Done. ({dt_inf:.1f}ms inf, {dt_nms:.1f}ms NMS)")

    # Mostra video
    cv2.imshow(WINDOW_NAME, final_frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()