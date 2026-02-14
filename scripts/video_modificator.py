import cv2
import os
from ultralytics import YOLO
from sys import argv
import torch

class VideoAnnotator:
    def __init__(self, video_path, model_path, output_dir):
        if not os.path.exists(video_path):
            print("Video not found")
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(video_path)
        if not os.path.exists(model_path):
            print("Model not found")
        self.model = YOLO(model_path, task="detect")
        if torch.cuda.is_available():
            self.model.to("cuda")
        
        # Classes
        self.CLASSES = {
            0: "Magikarp",
            1: "Patrat",
            2: "Binacle",
            3: "Kakuna",
            4: "Budew",
            5: "Staryu"
        }

        # Colors
        self.CLASS_COLORS = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 255, 0),  # Cyan
            (255, 0, 255)   # Magenta
        ]
        
        
        self.paused = False
        self.current_frame_id = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Box: [x1, y1, x2, y2, class_id, confidence]
        self.current_boxes = [] 
        self.selected_box = -1
        self.frame = None
        self.is_saved = False             
        self.consecutive_count = 0
        
        
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.temp_box = None

        # Output
        self.output_dir = output_dir + "/labels"
        os.makedirs(self.output_dir, exist_ok=True)

        # GUI Setup
        self.WINDOW_NAME = "Video Modificator"
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)

    def get_label_filename(self, frame_id=None):
        id = frame_id if frame_id is not None else self.current_frame_id
        return os.path.join(self.output_dir, f"{self.video_name.split('_')[2]}_frame_{id:06d}.txt")

    def update_consecutive_count(self):
        if not self.is_saved:
            self.consecutive_count = 0
            return

        frame_count = 1
        check_id = self.current_frame_id - 1
        while check_id >= 0:
            filename = self.get_label_filename(check_id)
            if os.path.exists(filename):
                frame_count += 1
                check_id -= 1
            else:
                break
        self.consecutive_count = frame_count

    def save_labels(self):
        filename = self.get_label_filename()
        
        if not self.current_boxes:
            if os.path.exists(filename):
                os.remove(filename)
            self.is_saved = False
            self.update_consecutive_count()
            return

        h, w, _ = self.frame.shape
        lines = []
        for box in self.current_boxes:
            x1, y1, x2, y2, cls, _ = box
            
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        
        self.is_saved = True
        self.update_consecutive_count()
        print(f"Saved: {filename}")

    def set_background(self):
        filename = self.get_label_filename()
        self.current_boxes = []
        self.selected_box = -1
        
        with open(filename, 'w') as f:
            pass 
            
        self.is_saved = True
        self.update_consecutive_count()
        print(f"Saved as background: {filename}")

    def confirm_and_advance(self):
        if self.current_boxes:
            self.save_labels()
        else:
            self.set_background()

        if self.current_frame_id < self.total_frames - 1:
            self.current_frame_id += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
            ret, self.frame = self.cap.read()
            if ret: 
                self.load_labels_if_exist()
        else:
            print("Last frame!")

    def load_labels_if_exist(self):
        filename = self.get_label_filename()
        self.current_boxes = []
        self.is_saved = False
        
        if os.path.exists(filename):
            self.is_saved = True
            h, w, _ = self.frame.shape
            with open(filename, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:5])
                
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)
                
                self.current_boxes.append([x1, y1, x2, y2, cls, None])
        else:
            self.is_saved = False
            results = self.model(self.frame, conf=0.2, verbose=False, half=False)
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                self.current_boxes.append([coords[0], coords[1], coords[2], coords[3], cls, conf])
        
        self.update_consecutive_count()

    def mouse_callback(self, event, x, y, _, __):
        if not self.paused:
            return 

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_box = False
            for i in range(len(self.current_boxes)-1, -1, -1):
                box = self.current_boxes[i]
                x1, y1, x2, y2, _, _ = box
                if x1 < x < x2 and y1 < y < y2:
                    self.selected_box = i
                    clicked_box = True
                    break
            
            if not clicked_box:
                self.drawing = True
                self.selected_box = -1
                self.ix, self.iy = x, y
            
            self.draw_interface()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.ix, self.iy, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.current_boxes.append([x1, y1, x2, y2, 0, None])
                    self.selected_box = len(self.current_boxes)-1
                    self.save_labels()
                self.temp_box = None
                self.draw_interface()

    def draw_interface(self):
        if self.frame is None:
            return
        
        display_frame = self.frame.copy()
        h, w, _ = display_frame.shape

        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2, cls, conf = box
            
            if i == self.selected_box:
                bg_color = (255, 255, 255)
                text_color = (0, 0, 0)
                thickness = 3
            else:
                bg_color = self.CLASS_COLORS[cls]
                thickness = 3
                b, g, r = bg_color
                luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                if luminance > 128:
                    text_color = (0, 0, 0)
                else:
                    text_color = (255, 255, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), bg_color, thickness)
            
            class_name = self.CLASSES.get(cls, f"ID {cls}")
            label_text = f"{class_name} {conf:.2f}" if conf is not None else f"{class_name}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            if y1 - text_size[1] - 5 < 0:
                text_org = (x1, y1 + text_size[1] + 5)
                rect_start = (x1, y1)
                rect_end = (x1 + text_size[0], y1 + text_size[1] + 10)
            else:
                text_org = (x1, y1 - 5)
                rect_start = (x1, y1 - text_size[1] - 10)
                rect_end = (x1 + text_size[0], y1)

            cv2.rectangle(display_frame, rect_start, rect_end, bg_color, -1)
            cv2.putText(display_frame, label_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        if self.temp_box:
            cv2.rectangle(display_frame, (self.temp_box[0], self.temp_box[1]), (self.temp_box[2], self.temp_box[3]), (200, 200, 200), 1)

        cv2.putText(display_frame, f"Frame: {self.current_frame_id}/{self.total_frames}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        if not self.is_saved:
             status_msg = "Not saved"
             status_color = (0, 165, 255) 
        else:
             status_msg = f"Saved {len(self.current_boxes)} box"
             status_color = (255, 255, 0) 
        cv2.putText(display_frame, status_msg, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if self.is_saved:
            text = f"[Saved {self.consecutive_count} consecutive frames]"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = w - text_size[0] - 20
            cv2.putText(display_frame, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        instr = "Play/Pause [Space] | Save+Next [s] | Del [x] | Select [Click] | Class [0-5] | Nav [A/D]"
        
        instr_size = cv2.getTextSize(instr, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]
        cv2.rectangle(display_frame, (5, h - 65), (15 + instr_size[0], h - 45), (0, 0, 0), -1)
        
        cv2.putText(display_frame, instr, (10, h - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
        cv2.imshow(self.WINDOW_NAME, display_frame)

    def run(self):
        while self.cap.isOpened():
            if not self.paused:
                success, frame = self.cap.read()
                if not success:
                    break 
                self.frame = frame
                self.current_frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                self.load_labels_if_exist()
            
            self.draw_interface()
            
            wait_time = 0 if self.paused else 1
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused

            if self.paused:
                if key == ord('d'): 
                    if self.current_frame_id < self.total_frames-1:
                        self.current_frame_id += 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
                        ret, self.frame = self.cap.read()
                        if ret: self.load_labels_if_exist()
                elif key == ord('a'): 
                    if self.current_frame_id > 0:
                        self.current_frame_id -= 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
                        ret, self.frame = self.cap.read()
                        if ret: self.load_labels_if_exist()
                
                elif key in [ord(str(i)) for i in range(6)]:
                    if self.selected_box != -1 and self.selected_box < len(self.current_boxes):
                        self.current_boxes[self.selected_box][4] = int(chr(key))
                        self.selected_box = -1 
                        self.save_labels()
                        self.draw_interface() 

                elif key == ord('x'):
                    if self.selected_box != -1 and self.selected_box < len(self.current_boxes):
                        self.current_boxes.pop(self.selected_box)
                        self.selected_box = -1 
                        self.save_labels()
                        self.draw_interface()
                
                elif key == ord('s'):
                    self.confirm_and_advance()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    if len(argv) < 4:
        print("python video_modificator.py <videos/test/Zona_2_09.mp4> <models/single.pt> <dataset_single>")
        exit(1)
    
    video_path = argv[1]
    model_path = argv[2]
    output_dir = argv[3]

    annotator = VideoAnnotator(video_path, model_path, output_dir)
    annotator.run()