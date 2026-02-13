import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from sys import argv

# --- 1. PATH SETUP & IMPORTS ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
YOLO_PATH = ROOT / "yolov5"
print(f"YOLO_PATH: {YOLO_PATH}")

if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# --- 2. LOAD MODEL (raw, no AutoShape wrapper) ---
print("Caricamento modello...")
device = select_device('0' if torch.cuda.is_available() else 'cpu')
weights_path = "yolov5\\runs\\train\\stream7\\weights\\last.pt"

model = DetectMultiBackend(str(weights_path), device=device)
stride = int(model.stride)
names = model.names
img_size = check_img_size(640, s=stride)
model.eval()

# --- 3. LOAD IMAGES ---
video_num = argv[1] if len(argv) > 1 else "09"
# T-1 = support frame, T = current frame, T+1 = ground truth (for visual comparison)
path_img_t_minus_1 = ROOT / "Test_predict" / f"{video_num}_frame_000000.jpg"
path_img_t         = ROOT / "Test_predict" / f"{video_num}_frame_000001.jpg"
path_img_output    = ROOT / "Test_predict" / f"{video_num}_frame_000002.jpg"

img_support_orig = cv2.imread(str(path_img_t_minus_1))  # T-1 (support)
img_current_orig = cv2.imread(str(path_img_t))           # T   (current)
img_next_orig    = cv2.imread(str(path_img_output))      # T+1 (visual reference)

if img_support_orig is None or img_current_orig is None or img_next_orig is None:
    print("Errore: Impossibile caricare le immagini.")
    sys.exit()

# --- 4. PRE-PROCESSING ---
def preprocess(img_bgr, img_size, stride):
    """Letterbox + BGR->RGB + HWC->CHW + normalize -> (1, 3, H, W) tensor."""
    img_lb = letterbox(img_bgr, img_size, stride=stride, auto=False)[0]
    img_rgb = img_lb[:, :, ::-1]  # BGR -> RGB
    img_chw = np.ascontiguousarray(img_rgb.transpose((2, 0, 1)))  # HWC -> CHW
    tensor = torch.from_numpy(img_chw).to(device).float() / 255.0
    return tensor.unsqueeze(0)  # (1, 3, H, W)

t_support = preprocess(img_support_orig, img_size, stride)  # (1, 3, 640, 640)
t_current = preprocess(img_current_orig, img_size, stride)  # (1, 3, 640, 640)

# Concatenate: 6ch = [current | support] — same order as training dataloader
img_6ch = torch.cat([t_current, t_support], dim=1)  # (1, 6, 640, 640)

# --- 5. INFERENCE (6ch -> DFP fuses current + support -> predicts T+1) ---
# Model sees (T, T-1) and predicts bounding boxes for T+1 (future frame)
print(f"Input shape: {img_6ch.shape}")
with torch.no_grad():
    pred = model(img_6ch)

# NMS
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
det = pred[0]
print(f"Detections: {len(det)}")

# --- 6. DRAW PREDICTIONS ON ALL THREE IMAGES ---
out1 = img_support_orig.copy()  # T-1
out2 = img_current_orig.copy()  # T
out3 = img_next_orig.copy()     # T+1 (predictions are FOR this frame)

if len(det):
    for i, target_img in enumerate([out1, out2, out3]):
        curr_det = det.clone()
        curr_det[:, :4] = scale_boxes(img_6ch.shape[2:], curr_det[:, :4], target_img.shape).round()

        annotator = Annotator(target_img, line_width=3, example=str(names))
        for *xyxy, conf, cls in reversed(curr_det):
            label = f'{names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))

        if i == 0: out1 = annotator.result()
        elif i == 1: out2 = annotator.result()
        elif i == 2: out3 = annotator.result()

# --- 7. DISPLAY COLLAGE ---
W_PANEL = 640
H_PANEL = 480
WINDOW_NAME = "StreamYOLO DFP: 6ch Inference"

v1 = cv2.resize(out1, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)
v2 = cv2.resize(out2, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)
v3 = cv2.resize(out3, (W_PANEL, H_PANEL), interpolation=cv2.INTER_AREA)

font = cv2.FONT_HERSHEY_DUPLEX
f_size = 0.8
cv2.putText(v1, "SUPPORT (T-1)", (15, 35), font, f_size, (255, 200, 0), 2)
cv2.putText(v2, "CURRENT (T) — INPUT", (15, 35), font, f_size, (0, 255, 0), 2)
cv2.putText(v3, "NEXT (T+1) — PREDICTED TARGET", (15, 35), font, f_size, (0, 0, 255), 2)

debug_collage = np.hstack((v1, v2, v3))

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.imshow(WINDOW_NAME, debug_collage)

print("Premi un tasto per chiudere.")
cv2.waitKey(0)
cv2.destroyAllWindows()
