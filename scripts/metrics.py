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
#ciao

def post_process(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Input:
        prediction: [1, N, 5 + num_classes]
    Output:
        List of [x1, y1, x2, y2, conf, cls]
    """
    
    if prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)

    # Remove by conf_thres
    mask = prediction[:, 4] > conf_thres
    det = prediction[mask]

    if det.shape[0] == 0:
        return torch.zeros((0, 6))

    scores, class_ids = det[:, 5:].max(1)
    det[:, 4] *= scores
    
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes = det[:, :4]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    new_boxes = torch.stack((x1, y1, x2, y2), dim=1)

    keep = nms(new_boxes, det[:, 4], iou_thres)

    result = torch.cat((new_boxes[keep], det[keep, 4:5], class_ids[keep].unsqueeze(1)), 1)
    return result


def letterbox_image(image, size=(640, 640)):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    
    new_image = Image.new('RGB', size, (114, 114, 114))
    
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image.paste(image, (dx, dy))
    
    return new_image, (dx, dy), scale

class StreamingDataset(Dataset):
    def __init__(self, root_dir, img_size=(640, 640), future_window=10):
        self.root_dir = root_dir
        self.img_size = img_size
        self.future_window = future_window
        
        self.img_dir = os.path.join(root_dir, 'images')
        self.lbl_dir = os.path.join(root_dir, 'labels')
        
        self.img_files = sorted([f for f in os.listdir(self.img_dir)])
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files) - self.future_window - 1

    def _load_yolo_labels(self, label_path, img_w, img_h):
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    bw = w * img_w
                    bh = h * img_h
                    boxes.append([x1, y1, bw, bh])
                    labels.append(int(cls))
        return boxes, labels

    def __getitem__(self, idx):
        path_prev = os.path.join(self.img_dir, self.img_files[idx])
        path_curr = os.path.join(self.img_dir, self.img_files[idx + 1])
        
        img_prev = Image.open(path_prev).convert("RGB")
        img_curr = Image.open(path_curr).convert("RGB")
        orig_w, orig_h = img_curr.size

        lb_prev, _, _ = letterbox_image(img_prev, self.img_size)
        lb_curr, pad, scale = letterbox_image(img_curr, self.img_size)

        t_prev = self.to_tensor(lb_prev)
        t_curr = self.to_tensor(lb_curr)

        future_gt = []
        for i in range(1, self.future_window + 1):
            f_idx = idx + 1 + i
            label_name = os.path.splitext(self.img_files[f_idx])[0] + '.txt'
            label_path = os.path.join(self.lbl_dir, label_name)
            boxes, labels = self._load_yolo_labels(label_path, orig_w, orig_h)
            future_gt.append({'boxes': boxes, 'labels': labels, 'width': orig_w, 'height': orig_h})
        
        meta = {'pad': pad, 'scale': scale, 'orig_shape': (orig_w, orig_h)}
        
        return t_prev, t_curr, future_gt, idx + 1, meta


def scale_coords(coords, pad, scale, img_shape):
    coords[:, 0] -= pad[0]  # x padding
    coords[:, 2] -= pad[0]
    coords[:, 1] -= pad[1]  # y padding
    coords[:, 3] -= pad[1]
    
    coords[:, :4] /= scale
    
    coords[:, 0].clamp_(0, img_shape[0])  # x1
    coords[:, 1].clamp_(0, img_shape[1])  # y1
    coords[:, 2].clamp_(0, img_shape[0])  # x2
    coords[:, 3].clamp_(0, img_shape[1])  # y2
    
    return coords

class SAPEvaluator:
    def __init__(self, categories):
        self.categories = categories
        self.dataset = {"images": [], "annotations": [], "categories": categories}
        self.predictions = []
        self.annotation_counter = 1

    def add_data(self, frame_id, gt, detections, meta):
        self.dataset["images"].append({
            "id": frame_id, 
            "width": gt['width'], 
            "height": gt['height']
        })

        for box, label in zip(gt['boxes'], gt['labels']):
            self.dataset["annotations"].append({
                "id": self.annotation_counter,
                "image_id": frame_id,
                "category_id": int(label),
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            })
            self.annotation_counter += 1

        if detections is not None and len(detections) > 0:
            det_scaled = detections.clone()

            pad = meta['pad']
            scale = meta['scale']
            orig_wh = meta['orig_shape']
            
            det_scaled[:, :4] = scale_coords(det_scaled[:, :4], pad, scale, orig_wh)

            for i in range(det_scaled.shape[0]):
                x1, y1, x2, y2, conf, cls = det_scaled[i].tolist()
                
                w = x2 - x1
                h = y2 - y1
                
                self.predictions.append({
                    "image_id": frame_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, w, h],
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


def run_streaming_eval(yolo_dir, weights_path, data_root, fps=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame_duration = 1.0 / fps

    model = torch.hub.load(yolo_dir, 'custom', path=weights_path, source='local')
    model = model.model
    model.to(device).eval()

    dataset = StreamingDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    evaluator = SAPEvaluator(CATEGORIES)

    print(f"Evaluation on {device}...")
    print(f'Frame num: {len(dataloader)}')


    with torch.no_grad():
        for t_prev, t_curr, future_gts, frame_id, meta in dataloader:
            start_time = time.perf_counter()
            
            input_tensor = torch.cat([t_curr, t_prev], dim=0).unsqueeze(0).to(device)
            
            raw_output = model(input_tensor)
            detections = post_process(raw_output)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time

            if frame_id % 100 == 0:
                print(f"Frame: {frame_id}/{len(dataloader)}",end="\r",flush=True)

            delay_frames = int(np.ceil(elapsed / frame_duration))
            shift = max(0, delay_frames - 1)

            if shift < len(future_gts):
                evaluator.add_data(frame_id, future_gts[shift], detections, meta)

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
