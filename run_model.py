import sys
from pathlib import Path
import time
import torch
import cv2
import numpy as np
from sys import argv


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLO_PATH = ROOT / "yolov5"
if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


video_path = argv[1]
weights_path = argv[2]


try:
    model = torch.hub.load(str(YOLO_PATH), 'custom', path=str(weights_path), source='local')
except Exception as e:
    print(f"Error on opening model: {e}")
    sys.exit(1)

model.to(device).eval()

print(f"Model opened on {device}")



cap = cv2.VideoCapture(video_path)
WINDOW_NAME = "Model Test"

img_size = 640
frame_count = 0

past_frame_tensor = None
pres_frame_tensor = None

print("Press 'q' to stop")


while cap.isOpened():
    t0 = time.time()
    
    success, frame = cap.read()
    if not success:
        print("Video ended or error")
        break
        
    frame_count += 1
    frame_future = frame.copy()

    # Resize image with letterbox
    img_resized = letterbox(frame_future, img_size, stride=32, auto=False)[0]

    # Changing color and axis
    img_numpy = img_resized.transpose((2, 0, 1))[::-1] 
    img_numpy = np.ascontiguousarray(img_numpy)
    fut_frame_tensor = torch.from_numpy(img_numpy).to(device)
    fut_frame_tensor = fut_frame_tensor.float() / 255.0

    if pres_frame_tensor is None:
        pres_frame_tensor = fut_frame_tensor
        continue
    
    if past_frame_tensor is None:
        support_tensor = pres_frame_tensor
    else:
        support_tensor = past_frame_tensor

    
    input_tensor = torch.cat((pres_frame_tensor, support_tensor), 0)

    # Add batch dimension (1, 6, 640, 640)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor[None]
    
    past_frame_tensor = pres_frame_tensor.clone()
    pres_frame_tensor = fut_frame_tensor.clone()

    t1 = time.time()


    with torch.no_grad():
        pred = model(input_tensor)
    
    t2 = time.time()

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)
    
    t3 = time.time()

    # Times
    dt_inf = (t2 - t1) * 1000
    dt_nms = (t3 - t2) * 1000


    det = pred[0]
    frame_summary = f"Frame {frame_count}: "
    
    annotator = Annotator(frame_future, line_width=2, example=str(model.names))

    if len(det):
        det[:, :4] = scale_boxes(input_tensor.shape[2:], det[:, :4], frame_future.shape).round()

        for cls in det[:, 5].unique():
            n = (det[:, 5] == cls).sum()
            frame_summary += f"{n} {model.names[int(cls)]}{'s' * (n > 1)}, "
        frame_summary = frame_summary[:-2]

        # Draw boxes
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{model.names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            
        final_frame = annotator.result()
    else:
        frame_summary += "no detections"
        final_frame = frame_future

    
    
    frame_summary += f' ({dt_inf:.1f}ms inf, {dt_nms:.1f}ms NMS)'

    cv2.putText(final_frame, frame_summary, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)

    print(frame_summary)

    cv2.imshow(WINDOW_NAME, final_frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()