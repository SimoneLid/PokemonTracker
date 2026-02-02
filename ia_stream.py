import cv2
import mss
import numpy as np
import threading
import time
import tkinter as tk
import win32gui
import win32con
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
MODEL_PATH = "models/single_v3.pt"
CONFIDENCE = 0.2

# Colore di sfondo che diventerà trasparente (un colore che NON userai mai per i box)
# Ho scelto un grigio scuro strano, ma va bene qualsiasi cosa non sia nella lista sotto.
TRANSPARENT_BG = '#010101' 

# I tuoi colori personalizzati (RGB)
CLASS_COLORS = [
    (0, 255, 0),    # 0 - Verde
    (0, 0, 255),    # 1 - Blu
    (255, 0, 0),    # 2 - Rosso
    (0, 255, 255),  # 3 - Ciano
    (255, 255, 0),  # 4 - Giallo
    (255, 0, 255)   # 5 - Magenta
]

def rgb_to_hex(rgb):
    """Converte una tupla (R, G, B) in stringa hex #RRGGBB per Tkinter"""
    return '#%02x%02x%02x' % rgb

class GameAI:
    def __init__(self):
        print(f"Caricamento modello YOLO da {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        print("Modello pronto.")

    def detect(self, frame):
        results = self.model(frame, conf=CONFIDENCE, verbose=False)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy()) # Indice numerico della classe (0, 1, 2...)
                label = result.names[cls_id]
                
                # Salviamo anche cls_id per scegliere il colore dopo
                detected_objects.append((x1, y1, x2, y2, label, conf, cls_id))
                
        return detected_objects

class SharedState:
    def __init__(self):
        self.current_frame = None
        self.detections = []
        self.running = True
        self.lock = threading.Lock()

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def update_detections(self, detections):
        self.detections = detections

# --- THREAD IA (BACKGROUND) ---
def ai_worker(ai_model, state):
    while state.running:
        frame = state.get_frame()
        if frame is not None:
            detections = ai_model.detect(frame)
            state.update_detections(detections)
        else:
            time.sleep(0.01)

# --- THREAD CATTURA SCHERMO (BACKGROUND) ---
def capture_worker(state):
    sct = mss.mss()
    monitor = sct.monitors[1]
    
    while state.running:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        state.update_frame(frame)
        # Piccolo sleep per non intasare la CPU se necessario
        # time.sleep(0.001) 

# --- GUI OVERLAY (MAIN THREAD) ---
class OverlayApp:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', TRANSPARENT_BG)
        
        # Canvas con sfondo del colore che diventerà invisibile
        self.canvas = tk.Canvas(root, bg=TRANSPARENT_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.make_click_through()
        self.update_overlay()

    def make_click_through(self):
        hwnd = win32gui.GetParent(self.root.winfo_id())
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)

    def update_overlay(self):
        if not self.state.running:
            self.root.destroy()
            return

        self.canvas.delete("all")
        detections = self.state.detections
        
        for (x1, y1, x2, y2, label, conf, cls_id) in detections:
            # Selezione colore sicuro (se cls_id > 5, ricomincia da 0 usando il modulo %)
            rgb_color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            hex_color = rgb_to_hex(rgb_color)
            
            # Disegna rettangolo
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=hex_color, width=2)
            
            # Disegna testo
            label_text = f"{label} {conf:.2f}"
            self.canvas.create_text(x1, y1-10, text=label_text, fill=hex_color, font=("Arial", 12, "bold"), anchor="w")

        self.root.after(16, self.update_overlay)

def main():
    state = SharedState()
    
    try:
        ai_model = GameAI()
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    t_ai = threading.Thread(target=ai_worker, args=(ai_model, state))
    t_ai.daemon = True
    t_ai.start()

    t_cap = threading.Thread(target=capture_worker, args=(state,))
    t_cap.daemon = True
    t_cap.start()

    print("Overlay avviato. Premi CTRL+C nella console per chiudere.")
    
    root = tk.Tk()
    # Imposta sfondo root per sicurezza
    root.config(bg=TRANSPARENT_BG)
    app = OverlayApp(root, state)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        state.running = False

if __name__ == "__main__":
    main()