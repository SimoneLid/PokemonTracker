import cv2
import mss
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import PhotoImage, ttk  # Aggiunto ttk per il menu a tendina
import win32gui
import win32con
from ultralytics import YOLO
from PIL import Image, ImageTk
import os  # Aggiunto per leggere i file
import glob

POKEMON_LIST = [
    "Magikarp", "Patrat", "Binacle", "Kakuna", "Budew", "Staryu"
]

ALLOWED_POKEMONS = list(True for _ in range(len(POKEMON_LIST)))


# --- CONFIGURAZIONE ---
MODELS_DIR = "models"  # Cartella dove cercare i modelli .pt
DEFAULT_MODEL = "single_v3.pt"
CONFIDENCE = 0.2
# Parola chiave da cercare nel titolo della finestra
TARGET_WINDOW_KEYWORD = "citron "

# Colori
TRANSPARENT_BG = '#010101'
PANEL_COLOR = '#000000'     # Nero solido per il rettangolo in basso a sinistra
MENU_BG_COLOR = '#222222'   # Colore sfondo menu in alto

# Colori personalizzati (RGB)
CLASS_COLORS = [
    (0, 255, 0),    # 0 - Verde
    (0, 0, 255),    # 1 - Blu
    (255, 0, 0),    # 2 - Rosso
    (0, 255, 255),  # 3 - Ciano
    (255, 255, 0),  # 4 - Giallo
    (255, 0, 255)   # 5 - Magenta
]


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def find_window_by_partial_title(partial_title):
    target_hwnd = None

    def callback(hwnd, _):
        nonlocal target_hwnd
        title = win32gui.GetWindowText(hwnd)
        if win32gui.IsWindowVisible(hwnd) and partial_title.lower() in title.lower():
            target_hwnd = hwnd
    win32gui.EnumWindows(callback, None)
    return target_hwnd


class GameAI:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        
        # Crea la cartella se non esiste
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            print(f"Creata cartella {MODELS_DIR}. Inserisci i file .pt qui.")

        # Cerca il modello di default o il primo disponibile
        initial_path = os.path.join(MODELS_DIR, DEFAULT_MODEL)
        if os.path.exists(initial_path):
            self.change_model(initial_path)
        else:
            # Fallback sul primo .pt trovato
            pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
            if pt_files:
                self.change_model(pt_files[0])
            else:
                print("NESSUN MODELLO TROVATO nella cartella 'models/'!")

    def change_model(self, model_path):
        """Carica un nuovo modello in modo thread-safe"""
        print(f"Caricamento modello: {model_path}...")
        try:
            new_model = YOLO(model_path)
            with self.lock:
                self.model = new_model
            print(f"Modello {os.path.basename(model_path)} caricato con successo.")
        except Exception as e:
            print(f"Errore caricamento modello: {e}")

    def detect(self, frame):
        with self.lock:
            current_model = self.model
        
        if current_model is None:
            return []

        results = current_model(frame, conf=CONFIDENCE, verbose=False)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                # Controllo bounds per evitare crash se il modello ha più classi della lista
                if cls_id < len(ALLOWED_POKEMONS):
                    if not ALLOWED_POKEMONS[cls_id]:
                        continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                label = result.names[cls_id]
                detected_objects.append((x1, y1, x2, y2, label, conf, cls_id))
        return detected_objects


class SharedState:
    def __init__(self):
        self.current_frame = None
        self.detections = []
        self.running = True
        self.lock = threading.Lock()
        self.window_rect = None

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def update_detections(self, detections):
        self.detections = detections

    def set_window_rect(self, rect):
        self.window_rect = rect

    def get_window_rect(self):
        return self.window_rect


def ai_worker(ai_model, state):
    while state.running:
        frame = state.get_frame()
        if frame is not None:
            detections = ai_model.detect(frame)
            state.update_detections(detections)
        else:
            time.sleep(0.01)


def capture_worker(state):
    sct = mss.mss()
    while state.running:
        hwnd = find_window_by_partial_title(TARGET_WINDOW_KEYWORD)
        if hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                x, y, x2, y2 = rect
                w = x2 - x
                h = y2 - y
                if w > 0 and h > 0:
                    monitor = {'top': y, 'left': x, 'width': w, 'height': h}
                    state.set_window_rect(monitor)
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    state.update_frame(frame)
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        else:
            time.sleep(2)

# --- GUI OVERLAY ---

class OverlayApp:
    def __init__(self, root, state, ai_model):
        self.root = root
        self.state = state
        self.ai_model = ai_model # Riferimento all'AI per cambiare modello

        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', TRANSPARENT_BG)

        self.canvas = tk.Canvas(root, bg=TRANSPARENT_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.setup_window_style()
        self.setup_bottom_panel() # Spostato logica bottoni qui
        self.setup_top_menu()     # NUOVO: Menu in alto a sinistra

        self.update_overlay()

    def setup_window_style(self):
        hwnd = win32gui.GetParent(self.root.winfo_id())
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               style | win32con.WS_EX_LAYERED)

    def setup_top_menu(self):
        """Crea il menu a tendina in alto a sinistra"""
        # Frame contenitore per dare uno sfondo
        top_frame = tk.Frame(self.root, bg=MENU_BG_COLOR, padx=5, pady=5)
        top_frame.place(x=0, y=0) # Attaccato al bordo in alto a sinistra

        # Etichetta
        lbl = tk.Label(top_frame, text="Model:", bg=MENU_BG_COLOR, fg="white", font=("Arial", 10))
        lbl.pack(side=tk.LEFT, padx=(0, 5))

        # Trova i file .pt
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
        if not model_files:
            model_files = ["No models found"]

        # Combobox
        self.model_combo = ttk.Combobox(top_frame, values=model_files, state="readonly", width=20)
        self.model_combo.pack(side=tk.LEFT)
        
        # Seleziona il modello corrente se presente nella lista
        current_model_name = DEFAULT_MODEL
        if current_model_name in model_files:
            self.model_combo.set(current_model_name)
        elif model_files:
            self.model_combo.current(0)

        # Evento cambio selezione
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

    def on_model_change(self, event):
        selected_model = self.model_combo.get()
        full_path = os.path.join(MODELS_DIR, selected_model)
        # Lancia il cambio modello in un thread separato per non bloccare la GUI
        threading.Thread(target=self.ai_model.change_model, args=(full_path,), daemon=True).start()

    def setup_bottom_panel(self):
        """Logica originale dei bottoni spostata qui per pulizia"""
        self.buttons = []
        self.button_images = []
        self.last_btn_size = 0 

        for i in range(6):
            btn_text = POKEMON_LIST[i] if i < len(POKEMON_LIST) else f"Class {i}"
            btn = tk.Button(
                self.root,
                text=btn_text,
                compound="top",
                bg="#005500",
                fg="white",
                activebackground="#444444",
                activeforeground="white",
                bd=0,
                relief="flat"
            )
            # Fix lambda variable capture
            btn.config(command=lambda idx=i, b=btn: self.on_button_click(idx, b))
            self.buttons.append(btn)

    def on_button_click(self, index, button):
        if index < len(ALLOWED_POKEMONS):
            ALLOWED_POKEMONS[index] = not ALLOWED_POKEMONS[index]
            if ALLOWED_POKEMONS[index]:
                button.config(bg="#005500")
            else:
                button.config(bg="#222222")

    def update_ui_positions(self, win_rect):
        if not win_rect:
            for btn in self.buttons:
                btn.place_forget()
            return

        wx, wy, ww, wh = win_rect['left'], win_rect['top'], win_rect['width'], win_rect['height']
        
        # --- PANNELLO INFERIORE (Bottoni) ---
        panel_w = int(ww * 0.30)
        panel_h = int(wh * 0.15)
        panel_x = wx
        panel_y = wy + wh - panel_h
        
        self.canvas.create_rectangle(
            panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
            fill=PANEL_COLOR, outline=PANEL_COLOR
        )

        num_buttons = len(self.buttons)
        padding = 5
        max_h = panel_h - (padding * 2)
        max_w = (panel_w - (padding * (num_buttons + 1))) // num_buttons
        btn_size = min(max_h, max_w)

        if btn_size != self.last_btn_size and btn_size > 0:
            self.last_btn_size = btn_size
            img_side = int(btn_size * 0.60)
            self.button_images.clear()
            for i, btn in enumerate(self.buttons):
                try:
                    # Controlla se l'immagine esiste
                    path = f"sprites/{i}.png"
                    if os.path.exists(path):
                        new_img = Image.open(path).resize((img_side, img_side))
                        img_tk = ImageTk.PhotoImage(new_img)
                        self.button_images.append(img_tk)
                        btn.config(image=img_tk)
                    else:
                        # Placeholder se l'immagine non esiste
                        self.button_images.append(None)
                        btn.config(image='') 
                except Exception:
                    self.button_images.append(None)

        total_content_width = (max_w * num_buttons) + (padding * (num_buttons - 1))
        start_x_offset = (panel_w - total_content_width) // 2
        start_y_offset = (panel_h - max_h) // 2

        current_x = panel_x + start_x_offset
        current_y = panel_y + start_y_offset

        for btn in self.buttons:
            btn.place(x=current_x, y=current_y, width=max_w, height=max_h)
            current_x += btn_size + padding

    def update_overlay(self):
        if not self.state.running:
            self.root.destroy()
            return

        self.canvas.delete("all")
        win_rect = self.state.get_window_rect()

        self.update_ui_positions(win_rect)

        if win_rect:
            offset_x = win_rect['left']
            offset_y = win_rect['top']
            
            # Disegna i rettangoli
            for (x1, y1, x2, y2, label, conf, cls_id) in self.state.detections:
                abs_x1, abs_y1 = x1 + offset_x, y1 + offset_y
                abs_x2, abs_y2 = x2 + offset_x, y2 + offset_y
                
                # Usa un colore di default se l'ID supera la lista colori
                color_idx = cls_id % len(CLASS_COLORS)
                hex_color = rgb_to_hex(CLASS_COLORS[color_idx])

                self.canvas.create_rectangle(
                    abs_x1, abs_y1, abs_x2, abs_y2, outline=hex_color, width=2)
                self.canvas.create_text(
                    abs_x1, abs_y1-10, text=f"{label} {conf:.2f}", 
                    fill=hex_color, font=("Arial", 12, "bold"), anchor="w")

        self.root.after(16, self.update_overlay)


def main():
    # Assicurati che esista la cartella models
    if not os.path.exists(MODELS_DIR):
        try:
            os.makedirs(MODELS_DIR)
        except:
            pass
            
    state = SharedState()
    ai_model = GameAI() # Inizializza l'AI

    t_ai = threading.Thread(target=ai_worker, args=(ai_model, state))
    t_ai.daemon = True
    t_ai.start()

    t_cap = threading.Thread(target=capture_worker, args=(state,))
    t_cap.daemon = True
    t_cap.start()

    print(f"In attesa della finestra '{TARGET_WINDOW_KEYWORD}'...")
    root = tk.Tk()
    root.config(bg=TRANSPARENT_BG)
    
    # Passiamo ai_model anche alla App GUI per poter cambiare modello
    app = OverlayApp(root, state, ai_model)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        state.running = False


if __name__ == "__main__":
    main()