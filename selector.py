import cv2
import os
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
# Assicurati che il percorso punti al modello corretto
MODEL_PATH = "runs\\detect\\runs\\micro\\autolabeler_model8\\weights\\best.pt"
VIDEO_PATH = "video_pokemon.mp4"

# Cartelle di destinazione
OUTPUT_BASE = "dataset_final"
OUTPUT_IMAGES = os.path.join(OUTPUT_BASE, "images")
OUTPUT_LABELS = os.path.join(OUTPUT_BASE, "labels")

# Crea cartelle se non esistono
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# Carica modello
print("Caricamento modello...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_count = 0

print("\n--- ISTRUZIONI ---")
print("MODALITÀ: BEST BOX ONLY (Solo il box con confidenza maggiore viene mostrato)")
print("Il video si ferma se trova qualcosa con conf > 0.20")
print(" [A] -> ACCETTA: Salva frame e l'unico box migliore")
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
    
    # Controlliamo se ha trovato qualcosa
    has_detection = False
    best_box = None

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        has_detection = True
        
        # --- LOGICA BEST BOX ---
        # Trova l'indice del box con la confidenza più alta
        best_idx = results[0].boxes.conf.argmax()
        
        # Estrai SOLO quel box specifico
        # Nota: results[0].boxes[best_idx] crea un nuovo oggetto Boxes contenente solo quello
        best_box = results[0].boxes[best_idx]
        
        # Sovrascriviamo i box nel risultato originale
        # Così quando chiamiamo .plot(), disegnerà solo questo!
        results[0].boxes = best_box

    # SE ha trovato qualcosa, mostriamo e chiediamo all'utente
    if has_detection:
        # Disegna (ora disegnerà solo il box migliore grazie alla sovrascrittura sopra)
        annotated_frame = results[0].plot()
        
        # Estraiamo la confidenza del "vincitore" per mostrarla
        current_conf = best_box.conf.item()
        
        # Info a schermo
        cv2.putText(annotated_frame, f"Best Conf: {current_conf:.2f} | Saved: {saved_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "[A] Accept  [D] Discard", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Supervisione Dataset (Best Box)", annotated_frame)

        # Blocca il flusso e aspetta input
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # --- TASTO 'A' (ACCETTA) ---
            if key == ord('a'):
                # 1. Salva immagine pulita (senza box disegnati)
                filename = f"frame_{frame_count:06d}"
                img_save_path = os.path.join(OUTPUT_IMAGES, filename + ".jpg")
                cv2.imwrite(img_save_path, frame)
                
                # 2. Salva Label txt (SOLO DEL BEST BOX)
                txt_save_path = os.path.join(OUTPUT_LABELS, filename + ".txt")
                with open(txt_save_path, 'w') as f:
                    # best_box contiene solo un elemento, quindi questo loop gira una volta sola
                    for box in best_box:
                        cls = int(box.cls[0])
                        x, y, w, h = box.xywhn[0].tolist()
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
                print(f"✅ Salvato frame {frame_count} (Conf: {current_conf:.2f})")
                saved_count += 1
                break 

            # --- TASTO 'D' (SCARTA) ---
            elif key == ord('d'):
                print(f"❌ Scartato frame {frame_count}")
                break 
            
            # --- TASTO 'Q' (ESCI) ---
            elif key == ord('q'):
                print("Chiusura script...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    else:
        # Se non trova nulla, passa avanti silenziosamente
        pass

cap.release()
cv2.destroyAllWindows()
print(f"\nFinito! Hai salvato {saved_count} immagini (Single-Instance) nel dataset.")