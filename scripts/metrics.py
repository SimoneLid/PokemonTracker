import os
import time
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CATEGORIES = [{"id": 0, "name": "Magikarp"},
              {"id": 1, "name": "Patrat"},
              {"id": 2, "name": "Binacle"},
              {"id": 3, "name": "Kakuna"},
              {"id": 4, "name": "Budew"},
              {"id": 5, "name": "Staryu"}]


def post_process(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Converte l'output grezzo [1, 25200, 85] in box filtrati.
    Args:
        prediction: Tensore [1, N, 5 + num_classes]
    Returns:
        List of [x1, y1, x2, y2, conf, cls]
    """
    # Rimuovi la dimensione batch se è 1
    if prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)

    # 1. Filtra per confidenza dell'oggetto (objectness)
    mask = prediction[:, 4] > conf_thres
    det = prediction[mask]

    if det.shape[0] == 0:
        return torch.zeros((0, 6))

    # 2. Calcola lo score finale (objectness * max_class_probability)
    scores, class_ids = det[:, 5:].max(1)
    det[:, 4] *= scores # Score finale
    
    # 3. Trasforma da [cx, cy, w, h] a [x1, y1, x2, y2]
    boxes = det[:, :4]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    new_boxes = torch.stack((x1, y1, x2, y2), dim=1)

    # 4. NMS (Non-Maximum Suppression)
    keep = nms(new_boxes, det[:, 4], iou_thres)
    
    # Output finale: [x1, y1, x2, y2, confidence, class_id]
    result = torch.cat((new_boxes[keep], det[keep, 4:5], class_ids[keep].unsqueeze(1)), 1)
    return result

# ==========================================
# 1. DATASET & LOADING UTILS
# ==========================================

class StreamingDataset(Dataset):
    def __init__(self, root_dir, img_size=(640, 640), future_window=10):
        self.root_dir = root_dir
        self.img_size = img_size
        self.future_window = future_window
        
        self.img_dir = os.path.join(root_dir, 'images')
        self.lbl_dir = os.path.join(root_dir, 'labels')
        
        # Carica e ordina i file per sequenza temporale
        self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                                if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files) - self.future_window - 1

    def _load_yolo_labels(self, label_path, img_w, img_h):
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    # Converti da YOLO (center_x, center_y, w, h) a COCO (x1, y1, w, h) in pixel
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    bw = w * img_w
                    bh = h * img_h
                    boxes.append([x1, y1, bw, bh])
                    labels.append(int(cls))
        return boxes, labels

    def __getitem__(self, idx):
        # Frame t-1 e t
        path_prev = os.path.join(self.img_dir, self.img_files[idx])
        path_curr = os.path.join(self.img_dir, self.img_files[idx + 1])
        
        img_prev = Image.open(path_prev).convert("RGB")
        img_curr = Image.open(path_curr).convert("RGB")
        orig_w, orig_h = img_curr.size

        # Trasformazione per il modello
        t_prev = self.transform(img_prev)
        t_curr = self.transform(img_curr)

        # Caricamento Ground Truth per i frame futuri
        future_gts = []
        for i in range(1, self.future_window + 1):
            f_idx = idx + 1 + i
            lbl_name = os.path.splitext(self.img_files[f_idx])[0] + '.txt'
            lbl_path = os.path.join(self.lbl_dir, lbl_name)
            boxes, labels = self._load_yolo_labels(lbl_path, orig_w, orig_h)
            future_gts.append({'boxes': boxes, 'labels': labels, 'width': orig_w, 'height': orig_h})

        return t_prev, t_curr, future_gts, idx + 1

# ==========================================
# 2. EVALUATOR CORE
# ==========================================

class SAPEvaluator:
    def __init__(self, categories):
        self.categories = categories
        self.dataset = {"images": [], "annotations": [], "categories": categories}
        self.predictions = []
        self.ann_id_counter = 1

    def add_data(self, frame_id, gt, detections, img_size=(640, 640)):
        """
        gt: dizionario con i ground truth
        detections: tensore [N, 6] -> [x1, y1, x2, y2, conf, cls]
        """
        # Aggiungi l'immagine al dataset COCO
        self.dataset["images"].append({
            "id": frame_id, 
            "width": gt['width'], 
            "height": gt['height']
        })

        # Aggiungi i Ground Truth
        for box, lbl in zip(gt['boxes'], gt['labels']):
            self.dataset["annotations"].append({
                "id": self.ann_id_counter,
                "image_id": frame_id,
                "category_id": int(lbl),
                "bbox": box, # [x, y, w, h]
                "area": box[2] * box[3],
                "iscrowd": 0
            })
            self.ann_id_counter += 1

        # Aggiungi le Predizioni
        # Calcoliamo il fattore di scala se l'immagine originale era diversa da 640x640
        gain_w = gt['width'] / img_size[0]
        gain_h = gt['height'] / img_size[1]

        # detections è un tensore [N, 6]
        for i in range(detections.shape[0]):
            det = detections[i]
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # Converti da [x1, y1, x2, y2] a [x, y, w, h] e scala su dimensione originale
            w = (x2 - x1) * gain_w
            h = (y2 - y1) * gain_h
            
            self.predictions.append({
                "image_id": frame_id,
                "category_id": int(cls),
                "bbox": [x1 * gain_w, y1 * gain_h, w, h],
                "score": float(conf)
            })

    def finalize(self):
        coco_gt = COCO()
        coco_gt.dataset = self.dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self.predictions)
        
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator.stats

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def run_streaming_eval(yolo_dir, weights_path, data_root, fps=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame_duration = 1.0 / fps

    # Carica Modello
    model = torch.hub.load(yolo_dir, 'custom', path=weights_path, source='local')
    model.to(device).eval()

    dataset = StreamingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    evaluator = SAPEvaluator(CATEGORIES)

    print(f"Inizio valutazione streaming su {device}...")
    print(f'Frame totali: {len(dataloader)}')


    with torch.no_grad():
        for t_prev, t_curr, future_gts, frame_id in dataloader:
            # --- Inizio Misurazione Latenza ---
            start_time = time.perf_counter()
            
            # Concatena frame come nel dataloader [t,t-1]
            input_tensor = torch.cat([t_curr, t_prev], dim=0).unsqueeze(0).to(device)
            
            raw_output = model(input_tensor) # Ritorna il tensore che hai mostrato
            
            # Se model() ritorna una lista/tuple (tipico di YOLO), prendi il primo elemento
            if isinstance(raw_output, (list, tuple)):
                raw_output = raw_output[0]

            detections = post_process(raw_output)

            # Sincronizza CUDA per misurare il tempo reale
            if device.type == 'cuda': torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            # --- Fine Misurazione ---
            if frame_id % 100 == 0:
                print(f"Frame: {frame_id}/{len(dataloader)}",end="\r",flush=True)

            # Logica Streaming: calcola quale frame futuro è "attuale" ora
            delay_frames = int(np.ceil(elapsed / frame_duration))
            # Se delay_frames=1 -> t+1, delay_frames=2 -> t+2, ecc.
            shift = max(0, delay_frames - 1)

            if shift < len(future_gts):
                evaluator.add_data(frame_id, future_gts[shift], detections)

    # Risultati Finali
    stats = evaluator.finalize()
    header = "weight_name,AP_all,AP_50,AP_75,AP_small,AP_medium,AP_large,AR_max1,AR_max10,AR_max100,AR_small,AR_medium,AR_large\n"
    
    if not os.path.exists("sAP.csv"):
        with open("sAP.csv", "w") as F:
            F.write(header)
    weights_parts = os.path.normpath(weights_path)
    weight_list = weights_parts.split(os.sep)
    stats_values = ",".join([f"{s:.4f}" for s in stats])
    with open("sAP.csv", "a") as F:
        F.write(f"{weight_list[-3]}_{weight_list[-1]},{stats_values}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weight.pt')
    parser.add_argument('--data', type=str, help='dataset path')
    args = parser.parse_args()

    run_streaming_eval(
        yolo_dir="yolov5",
        weights_path=args.weights, 
        data_root=args.data,
        fps=30
    )