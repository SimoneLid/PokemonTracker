import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# --- 1. CONFIGURAZIONE PATH E IMPORT ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
YOLO_PATH = ROOT / "yolov5"
print(f"YOLO_PATH: {YOLO_PATH}")

if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

# --- 2. CARICAMENTO MODELLO ---
print("Caricamento modello...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = YOLO_PATH / "runs" / "train" / "exp29" / "weights" / "best.pt" 

model = torch.hub.load(str(YOLO_PATH), 'custom', path=str(weights_path), source='local')
model.to(device).eval()

# --- 3. CARICAMENTO IMMAGINI ---
path_img_t_minus_1 = "Test_predict/frame_000067_Zona_2_02.jpg"
path_img_t = "Test_predict/frame_000068_Zona_2_02.jpg"
path_img_output = "Test_predict/frame_000069_Zona_2_02.jpg"

img1 = cv2.imread(path_img_t_minus_1)
img2 = cv2.imread(path_img_t)
img3 = cv2.imread(path_img_output)

if img1 is None or img2 is None or img3 is None:
    print("Errore: Impossibile caricare le immagini.")
    sys.exit()

# --- 4. PRE-PROCESSING (6 CANALI) ---
img_size = 640
img1_res = letterbox(img1, img_size, stride=32, auto=False)[0]
img2_res = letterbox(img2, img_size, stride=32, auto=False)[0]

img_tensor = img1_res.transpose((2, 0, 1))[::-1] 
img_tensor = np.ascontiguousarray(img_tensor)
img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
img_tensor = img_tensor[None]

# --- 5. INFERENZA ---
with torch.no_grad():
    pred = model(img_tensor)

pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
det = pred[0]

img_tensor = img2_res.transpose((2, 0, 1))[::-1] 
img_tensor = np.ascontiguousarray(img_tensor)
img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
img_tensor = img_tensor[None]

with torch.no_grad():
    pred = model(img_tensor)

pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
det = pred[0]
# --- 6. DISEGNO SU TUTTE E TRE LE IMMAGINI ---
# Creiamo tre copie separate per l'output
out1, out2, out3 = img1.copy(), img2.copy(), img3.copy()

if len(det):
    # Dobbiamo riscalare i box per ogni immagine (nel caso abbiano risoluzioni diverse)
    # Copiamo le det per non sovrascrivere le coordinate originali durante il loop
    for i, target_img in enumerate([out1, out2, out3]):
        # Riscaliamo i box sulle dimensioni dell'immagine corrente
        curr_det = det.clone()
        curr_det[:, :4] = scale_boxes(img_tensor.shape[2:], curr_det[:, :4], target_img.shape).round()
        
        annotator = Annotator(target_img, line_width=3, example=str(model.names))
        for *xyxy, conf, cls in reversed(curr_det):
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
        
        # Aggiorniamo l'immagine con i disegni
        if i == 0: out1 = annotator.result()
        elif i == 1: out2 = annotator.result()
        elif i == 2: out3 = annotator.result()

# --- 7. COLLAGE OTTIMIZZATO (640x480) ---
W_PANEL = 640 
H_PANEL = 480 
WINDOW_NAME = "6-CHANNELS DEBUG: LABELS ON ALL IMAGES"

# Ridimensionamento
v1 = cv2.resize(out1, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)
v2 = cv2.resize(out2, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)
v3 = cv2.resize(out3, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)

# Testi informativi
font = cv2.FONT_HERSHEY_DUPLEX
f_size = 0.8
cv2.putText(v1, "IMG 1 (T-1)", (15, 35), font, f_size, (255, 255, 255), 2)
cv2.putText(v2, "IMG 2 (T)", (15, 35), font, f_size, (255, 255, 255), 2)
cv2.putText(v3, "IMG 3 (NEXT)", (15, 35), font, f_size, (0, 0, 255), 2)

# Unione orizzontale
debug_collage = np.hstack((v1, v2, v3))

# --- 8. VISUALIZZAZIONE ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.imshow(WINDOW_NAME, debug_collage)

print("Visualizzazione completa. Premi un tasto per chiudere.")
cv2.waitKey(0)
cv2.destroyAllWindows()