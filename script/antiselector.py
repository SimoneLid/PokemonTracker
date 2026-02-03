import cv2
import os
from ultralytics import YOLO
from sys import argv

# --- CONFIGURAZIONE ---
if len(argv) < 4:
    print("Utilizzo: python script.py modello.pt video.mp4 output_cartella")
    exit()

MODEL_PATH = argv[1]
VIDEO_PATH = argv[2]
video_name = os.path.basename(VIDEO_PATH).rsplit(".")[0]

OUTPUT_BASE = argv[3]
OUTPUT_IMAGES = os.path.join(OUTPUT_BASE, "images")
OUTPUT_LABELS = os.path.join(OUTPUT_BASE, "labels")

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

print("Caricamento modello...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = -1
saved_count = 0

print("\n--- MODALITÀ: ESTRAZIONE BACKGROUND ---")
print("Il video si ferma SOLO quando NON viene rilevato nulla (conf > 0.20)")
print(" [A] -> ACCETTA: Salva frame e crea LABEL VUOTA (Background)")
print(" [D] -> SCARTA: Ignora e vai avanti")
print(" [Q] -> ESCI")
print("------------------\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # Eseguiamo la predizione
    results = model.predict(frame, conf=0.20, verbose=False)
    
    # Controlliamo se NON ha trovato nulla
    # Se len(results[0].boxes) == 0, significa che è potenziale background
    if results[0].boxes is None or len(results[0].boxes) == 0:
        
        display_frame = frame.copy()
        
        cv2.putText(display_frame, f"BACKGROUND Candidate - Frame {frame_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "[A] Save as Background  [D] Discard", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Supervisione Background", display_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):
                filename = f"frame_{frame_count:06d}_{video_name}"
                
                # 1. Salva immagine
                img_save_path = os.path.join(OUTPUT_IMAGES, filename + ".jpg")
                #cv2.imwrite(img_save_path, frame)
                
                # 2. Crea file label VUOTO (fondamentale per YOLO)
                txt_save_path = os.path.join(OUTPUT_LABELS, filename + ".txt")
                open(txt_save_path, 'a').close() # Crea un file a 0 byte
                
                print(f"Salvato Background: {filename}")
                saved_count += 1
                break 

            elif key == ord('d'):
                break 
            
            elif key == ord('q'):
                print("Chiusura script...")
                cap.release()
                cv2.destroyAllWindows()
                exit()
    else:
        # Se vede qualcosa, passa avanti (stiamo cercando solo i vuoti)
        pass

cap.release()
cv2.destroyAllWindows()
print(f"\nFinito! Hai aggiunto {saved_count} immagini di background al tuo dataset.")