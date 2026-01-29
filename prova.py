import cv2
from ultralytics import YOLO
import os
import glob

model = YOLO("yolov8n.pt")
output_folder = "dataset/val/labels"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join("dataset/val/images", "*.*")))


for image_name in image_paths:
    frame=cv2.imread(image_name)

    results = model(frame)
    
    # Nome del file senza estensione
    basename = os.path.basename(image_name)
    file_id = os.path.splitext(basename)[0]
    
    txt_filename = os.path.join(output_folder, f"{file_id}.txt")
    
    with open(txt_filename, "w") as f:
        # Usiamo .xywhn che sta per: x_center y_center width height
        max_conf=0
        boxes = results[0].boxes
        for i in range(len(boxes)):
            if boxes.conf[i]>max_conf:
                max_conf=boxes.conf[i]

                # Dati del box
                x, y, w, h = boxes.xywhn[i].tolist()
                class_id = int(boxes.cls[i].item())
        
        f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    annotated_frame = results[0].plot()


    cv2.imshow("Pokemon Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
