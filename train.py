from multiprocessing import freeze_support
from ultralytics import YOLO
from sys import argv
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        freeze_support()

        model = YOLO(argv[1]) 

        results = model.train(
            data=argv[2],
            epochs=50,
            imgsz=640,
            plots=True,
            device="cuda",
            batch=16,
            workers=8,
        )

    else:
        print("Cuda non presente sulla macchina")