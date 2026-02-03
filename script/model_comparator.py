import cv2
import os
from sys import argv
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
path_model_1 = argv[1]
path_model_2 = argv[2]

NAME_MODEL_1 = os.path.basename(path_model_1) 
NAME_MODEL_2 = os.path.basename(path_model_2)

video_path = argv[3]
WINDOW_NAME = "Confronto Modelli"
SCALE_FACTOR = 0.5 
# ----------------------

model1 = YOLO(path_model_1)
model2 = YOLO(path_model_2)

cap = cv2.VideoCapture(video_path)

# Impostazioni Font
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 0, 0) # Nero
THICKNESS = 2
MARGIN = 20 # Margine dal bordo

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Inferenza
        results1 = model1(frame, conf=0.2, verbose=False)
        results2 = model2(frame, conf=0.2, verbose=False)

        # --- FILTRO LOGICO ---
        if len(results1[0].boxes) == 0 and len(results2[0].boxes) == 0:
            continue 
        # ---------------------

        annotated1 = results1[0].plot()
        annotated2 = results2[0].plot()

        # Ridimensionamento
        height, width = annotated1.shape[:2]
        new_dim = (int(width * SCALE_FACTOR), int(height * SCALE_FACTOR))
        
        resized1 = cv2.resize(annotated1, new_dim)
        resized2 = cv2.resize(annotated2, new_dim)

        # --- CALCOLO POSIZIONE TESTO (Basso a Destra) ---
        
        # 1. Calcolo dimensione testo per Modello 1
        (text_w1, text_h1), _ = cv2.getTextSize(NAME_MODEL_1, FONT, FONT_SCALE, THICKNESS)
        # Coordinate: (LarghezzaImmagine - LarghezzaTesto - Margine, AltezzaImmagine - Margine)
        pos_x1 = new_dim[0] - text_w1 - MARGIN
        pos_y1 = new_dim[1] - MARGIN
        
        cv2.putText(resized1, NAME_MODEL_1, (pos_x1, pos_y1), 
                    FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

        # 2. Calcolo dimensione testo per Modello 2
        (text_w2, text_h2), _ = cv2.getTextSize(NAME_MODEL_2, FONT, FONT_SCALE, THICKNESS)
        pos_x2 = new_dim[0] - text_w2 - MARGIN
        pos_y2 = new_dim[1] - MARGIN

        cv2.putText(resized2, NAME_MODEL_2, (pos_x2, pos_y2), 
                    FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)
        # -----------------------------------------------

        # Unione e visualizzazione
        combined_view = cv2.hconcat([resized1, resized2])
        cv2.imshow(WINDOW_NAME, combined_view)

        # Loop attesa input
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("a"):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()
    else:
        break

cap.release()
cv2.destroyAllWindows()