import cv2
import mss
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import PhotoImage
import win32gui
import win32con
from ultralytics import YOLO
from PIL import Image, ImageTk

POKEMON_LIST = [
    "Magikarp", "Patrat", "Binacle", "Kakuna", "Budew", "Staryu"
]

ALLOWED_POKEMONS = list(False for _ in range(len(POKEMON_LIST)))


# --- CONFIGURAZIONE ---
MODEL_PATH = "models/single_v3.pt"
CONFIDENCE = 0.2
# Parola chiave da cercare nel titolo della finestra
TARGET_WINDOW_KEYWORD = "Movies"

# Colori
TRANSPARENT_BG = '#010101'
PANEL_COLOR = '#000000'     # Nero solido per il rettangolo in basso a sinistra

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
        print(f"Caricamento modello YOLO da {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH) # Scommenta se hai il modello
        print("Modello pronto.")

    def detect(self, frame):
        results = self.model(frame, conf=CONFIDENCE, verbose=False)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy()) # Indice numerico della classe (0, 1, 2...)
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
    def __init__(self, root, state):
        self.root = root
        self.state = state

        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', TRANSPARENT_BG)

        self.canvas = tk.Canvas(root, bg=TRANSPARENT_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.setup_window_style()

        self.buttons = []
        self.button_images = []
        self.last_btn_size = 0  # Traccia la dimensione precedente per evitare refresh inutili

        # Creazione iniziale bottoni (senza immagini per ora, verranno aggiunte dinamicamente)
        for i in range(6):
            btn = tk.Button(
                root,
                text=POKEMON_LIST[i],
                compound="top",  # Immagine sopra, testo sotto
                bg="#222222",
                fg="white",
                activebackground="#444444",
                activeforeground="white",
                bd=0,
                relief="flat"
            )
            btn.config(command=lambda idx=i, b=btn: self.on_button_click(idx, b))
            self.buttons.append(btn)

        self.update_overlay()

    def setup_window_style(self):
        hwnd = win32gui.GetParent(self.root.winfo_id())
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               style | win32con.WS_EX_LAYERED)

    def on_button_click(self, index, button):
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

        # 1. Calcolo Pannello Nero (25% W, 10% H)
        panel_w = int(ww * 0.25)
        panel_h = int(wh * 0.10)
        panel_x = wx
        panel_y = wy + wh - panel_h

        self.canvas.create_rectangle(
            panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
            fill=PANEL_COLOR, outline=PANEL_COLOR
        )

        # 2. Calcolo dimensione bottoni (Quadrati)
        num_buttons = len(self.buttons)
        padding = 5
        max_h = panel_h - (padding * 2)
        max_w = (panel_w - (padding * (num_buttons + 1))) // num_buttons
        btn_size = min(max_h, max_w)

        # 3. Aggiorna Immagini SOLO se la dimensione è cambiata
        #    Questo assicura che l'immagine sia sempre il 60% del bottone
        if btn_size != self.last_btn_size and btn_size > 0:
            self.last_btn_size = btn_size

            # Calcolo 60% della dimensione del bottone
            img_side = int(btn_size * 0.60)

            # Rigenera le immagini
            self.button_images.clear()
            for i, btn in enumerate(self.buttons):
                # Se usi immagini vere, qui useresti PIL: Image.open(...).resize((img_side, img_side))
                new_img = Image.open(f"sprites/{i}.png").resize((img_side, img_side))
                img_tk = ImageTk.PhotoImage(new_img)
                self.button_images.append(img_tk)
                btn.config(image=img_tk)

        # 4. Posiziona i bottoni
        total_content_width = (btn_size * num_buttons) + \
            (padding * (num_buttons - 1))
        start_x_offset = (panel_w - total_content_width) // 2
        start_y_offset = (panel_h - btn_size) // 2

        current_x = panel_x + start_x_offset
        current_y = panel_y + start_y_offset

        for btn in self.buttons:
            btn.place(x=current_x, y=current_y,
                      width=btn_size, height=btn_size)
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
            for (x1, y1, x2, y2, label, conf, cls_id) in self.state.detections:
                abs_x1, abs_y1 = x1 + offset_x, y1 + offset_y
                abs_x2, abs_y2 = x2 + offset_x, y2 + offset_y
                hex_color = rgb_to_hex(
                    CLASS_COLORS[cls_id % len(CLASS_COLORS)])

                self.canvas.create_rectangle(
                    abs_x1, abs_y1, abs_x2, abs_y2, outline=hex_color, width=2)
                self.canvas.create_text(
                    abs_x1, abs_y1-10, text=f"{label} {conf:.2f}", fill=hex_color, font=("Arial", 12, "bold"), anchor="w")

        self.root.after(16, self.update_overlay)


def main():
    state = SharedState()
    try:
        ai_model = GameAI()
    except Exception:
        pass

    t_ai = threading.Thread(target=ai_worker, args=(ai_model, state))
    t_ai.daemon = True
    t_ai.start()

    t_cap = threading.Thread(target=capture_worker, args=(state,))
    t_cap.daemon = True
    t_cap.start()

    print(f"In attesa della finestra '{TARGET_WINDOW_KEYWORD}'...")
    root = tk.Tk()
    root.config(bg=TRANSPARENT_BG)
    app = OverlayApp(root, state)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        state.running = False


if __name__ == "__main__":
    main()
