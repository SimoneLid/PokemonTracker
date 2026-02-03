import cv2
import os
import numpy as np
from ultralytics import YOLO

class VideoAnnotator:
    def __init__(self, video_path, model_path):
        # --- Configurazione Base ---
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        
        # --- Mappatura Classi ---
        self.CLASS_NAMES = {
            0: "Magikarp",
            1: "Patrat",
            2: "Binacle",
            3: "Kakuna",
            4: "Budew",
            5: "Staryu"
        }

        # --- Colori Specifici (BGR per OpenCV) ---
        self.CLASS_COLORS = [
            (0, 255, 0),    # 0 - Green
            (0, 0, 255),    # 1 - Red
            (255, 0, 0),    # 2 - Blue
            (0, 255, 255),  # 3 - Yellow
            (255, 255, 0),  # 4 - Cyan
            (255, 0, 255)   # 5 - Magenta
        ]
        
        # --- Stato del Player ---
        self.paused = False
        self.current_frame_idx = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- Dati Annotazioni ---
        # Formato: [x1, y1, x2, y2, class_id, confidence]
        self.current_boxes = [] 
        self.selected_box_idx = -1
        self.frame = None
        self.is_manual_background = False # Se True, il frame è confermato vuoto
        self.is_saved = False             # Se True, esiste un file .txt su disco
        
        # --- Variabili Mouse ---
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.temp_box = None

        # --- Output ---
        self.output_dir = "labels_dataset"
        os.makedirs(self.output_dir, exist_ok=True)

        # --- GUI Setup ---
        self.WINDOW_NAME = "Supervisione Dataset"
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)

    def get_label_filename(self):
        """Genera il percorso del file label per il frame corrente."""
        return os.path.join(self.output_dir, f"frame_{self.current_frame_idx:06d}_{self.video_name}.txt")

    def save_labels(self):
        """Salva i box su file. Se la lista è vuota, rimuove il file (stato Pending)."""
        filename = self.get_label_filename()
        
        # Se non ci sono box, cancelliamo il file per tornare allo stato "non supervisionato"
        # (Nota: Per salvare un frame vuoto intenzionale si usa set_background/tasto 'b')
        if not self.current_boxes:
            if os.path.exists(filename):
                os.remove(filename)
            self.is_manual_background = False
            self.is_saved = False
            return

        h, w, _ = self.frame.shape
        lines = []
        for box in self.current_boxes:
            x1, y1, x2, y2, cls, conf = box
            
            # Conversione YOLO (normalizzata 0-1)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Clamping
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        
        self.is_manual_background = False
        self.is_saved = True
        print(f"Salvato: {filename}")

    def set_background(self):
        """Crea forzatamente un file vuoto (Frame Background)."""
        filename = self.get_label_filename()
        self.current_boxes = []
        self.selected_box_idx = -1
        
        with open(filename, 'w') as f:
            pass # Crea file vuoto
            
        self.is_manual_background = True
        self.is_saved = True
        print(f"Salvato come BACKGROUND: {filename}")

    def confirm_and_advance(self):
        """Logica tasto 's': Salva lo stato attuale e vai al frame successivo."""
        if self.current_boxes:
            self.save_labels()
        else:
            # Se premo S e non ci sono box, assumo sia un background confermato
            self.set_background()

        # Avanza frame
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, self.frame = self.cap.read()
            if ret: 
                self.load_labels_if_exist()
        else:
            print("Ultimo frame raggiunto!")

    def load_labels_if_exist(self):
        """
        Carica i label se il file esiste.
        Se il file non esiste -> Esegue YOLO.
        Se il file esiste ma è vuoto -> Background (niente YOLO).
        """
        filename = self.get_label_filename()
        self.current_boxes = [] 
        self.is_manual_background = False
        self.is_saved = False
        
        if os.path.exists(filename):
            self.is_saved = True
            h, w, _ = self.frame.shape
            with open(filename, "r") as f:
                lines = f.readlines()
            
            # Controllo file vuoto (Background manuale)
            if not lines or all(l.strip() == '' for l in lines):
                self.is_manual_background = True
                return

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # Converti da YOLO (relativo) a Pixel (assoluto)
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    # Confidenza None perché caricato da file
                    self.current_boxes.append([x1, y1, x2, y2, cls, None])
        else:
            # File non esiste: Stato Pending -> Esegui YOLO
            self.is_saved = False
            results = self.model(self.frame, conf=0.2, verbose=False)
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                self.current_boxes.append([coords[0], coords[1], coords[2], coords[3], cls, conf])

    def mouse_callback(self, event, x, y, flags, param):
        """Gestisce click (selezione) e trascinamento (nuovo box)."""
        if not self.paused: return 

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Controlla click su box esistente
            clicked_box = False
            # Iteriamo al contrario per selezionare quelli "davanti"
            for i in range(len(self.current_boxes) - 1, -1, -1):
                box = self.current_boxes[i]
                x1, y1, x2, y2, _, _ = box
                if x1 < x < x2 and y1 < y < y2:
                    self.selected_box_idx = i
                    clicked_box = True
                    break
            
            if not clicked_box:
                # 2. Click sul vuoto -> Inizio disegno
                self.drawing = True
                self.selected_box_idx = -1
                self.ix, self.iy = x, y
            
            self.draw_interface()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.ix, self.iy, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Normalizza coordinate
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                
                # Ignora click involontari (box piccolissimi)
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    # Nuovo box: Classe 0 default, Confidenza None
                    self.current_boxes.append([x1, y1, x2, y2, 0, None])
                    self.selected_box_idx = len(self.current_boxes) - 1
                    self.save_labels()
                self.temp_box = None
                self.draw_interface()

    def draw_interface(self):
        """Disegna box, label adattive e stato interfaccia."""
        if self.frame is None: return
        
        display_frame = self.frame.copy()
        h, w, _ = display_frame.shape

        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2, cls, conf = box
            
            # --- SELEZIONE & COLORI ---
            if i == self.selected_box_idx:
                # SELEZIONATO: Box Bianco, Testo Nero, Spesso
                bg_color = (255, 255, 255) 
                text_color = (0, 0, 0)
                thickness = 3
            else:
                # NORMALE: Colore Classe
                safe_cls = cls if cls < len(self.CLASS_COLORS) else 0
                bg_color = self.CLASS_COLORS[safe_cls]
                thickness = 2
                
                # Calcolo Luminosità per colore testo (Formula standard)
                # BGR: B=0, G=1, R=2
                b, g, r = bg_color
                luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                # Se sfondo chiaro (>128) -> testo nero, altrimenti bianco
                text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

            # Disegna Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), bg_color, thickness)
            
            # --- ETICHETTA ---
            class_name = self.CLASS_NAMES.get(cls, f"ID {cls}")
            label_text = f"{class_name} {conf:.2f}" if conf is not None else f"{class_name}"
            
            t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Calcolo posizione etichetta (gestione bordo superiore)
            if y1 - t_size[1] - 5 < 0:
                # Disegna DENTRO il box se siamo al bordo superiore
                text_org = (x1, y1 + t_size[1] + 5)
                rect_start = (x1, y1)
                rect_end = (x1 + t_size[0], y1 + t_size[1] + 10)
            else:
                # Disegna SOPRA il box normalmente
                text_org = (x1, y1 - 5)
                rect_start = (x1, y1 - t_size[1] - 10)
                rect_end = (x1 + t_size[0], y1)

            # Sfondo pieno per il testo (Stesso colore del box)
            cv2.rectangle(display_frame, rect_start, rect_end, bg_color, -1)
            
            # Testo
            cv2.putText(display_frame, label_text, text_org, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Disegna box temporaneo (drag)
        if self.temp_box:
            cv2.rectangle(display_frame, (self.temp_box[0], self.temp_box[1]), 
                          (self.temp_box[2], self.temp_box[3]), (200, 200, 200), 1)

        # --- UI INFO OVERLAY ---
        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}/{self.total_frames}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Stato in basso a sinistra
        if self.is_manual_background:
            status_msg = "STATO: BACKGROUND (Vuoto confermato)"
            status_color = (0, 255, 0)
        elif not self.is_saved:
             status_msg = "STATO: PENDING (Non salvato)"
             status_color = (0, 165, 255) # Arancione
        else:
             status_msg = f"STATO: {len(self.current_boxes)} BOX OK"
             status_color = (255, 255, 0) # Ciano
        cv2.putText(display_frame, status_msg, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Indicatore SALVATO in alto a destra
        if self.is_saved:
            text = "[ SALVATO ]"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = w - text_size[0] - 20
            cv2.putText(display_frame, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Istruzioni in Pausa
        if self.paused:
             instr = "[s] Save&Next | [Click] Sel | [x] Del | [b] Bg | [0-5] Cls | [A/D] Nav | [Space] Play"
             cv2.putText(display_frame, instr, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow(self.WINDOW_NAME, display_frame)

    def run(self):
        while self.cap.isOpened():
            if not self.paused:
                success, frame = self.cap.read()
                if not success: break 
                self.frame = frame
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.load_labels_if_exist()
            
            self.draw_interface()
            
            # Gestione Tasti
            wait_time = 0 if self.paused else 1
            key = cv2.waitKey(wait_time) & 0xFF

            # -- Tasti Globali --
            if key == ord('q'): break
            elif key == ord(' '): self.paused = not self.paused

            # -- Tasti in Pausa --
            if self.paused:
                # Navigazione
                if key == ord('d'): 
                    if self.current_frame_idx < self.total_frames - 1:
                        self.current_frame_idx += 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                        ret, self.frame = self.cap.read()
                        if ret: self.load_labels_if_exist()
                elif key == ord('a'): 
                    if self.current_frame_idx > 0:
                        self.current_frame_idx -= 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                        ret, self.frame = self.cap.read()
                        if ret: self.load_labels_if_exist()
                
                # Cambio Classe (0-5)
                elif key in [ord(str(i)) for i in range(6)]:
                    if self.selected_box_idx != -1 and self.selected_box_idx < len(self.current_boxes):
                        # Assegna classe
                        self.current_boxes[self.selected_box_idx][4] = int(chr(key))
                        # Deseleziona e salva
                        self.selected_box_idx = -1 
                        self.save_labels()
                        self.draw_interface() 

                # Elimina Box
                elif key == ord('x'):
                    if self.selected_box_idx != -1 and self.selected_box_idx < len(self.current_boxes):
                        self.current_boxes.pop(self.selected_box_idx)
                        self.selected_box_idx = -1 
                        self.save_labels()
                        self.draw_interface()
                
                # Background Manuale
                elif key == ord('b'):
                    self.set_background()
                    self.draw_interface()
                
                # Salva e Avanti
                elif key == ord('s'):
                    self.confirm_and_advance()

        self.cap.release()
        cv2.destroyAllWindows()

# --- Avvio ---
if __name__ == "__main__":
    # Sostituisci con i tuoi percorsi corretti
    video_path = "videos/test/Zona_2_09.mp4"
    model_path = "models/single_v3.pt"
    
    annotator = VideoAnnotator(video_path, model_path)
    annotator.run()