from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == "__main__":
    freeze_support()

    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="pokemon.yaml",
        epochs=50,
        imgsz=640,
        plots=True,
        device="cuda",
        batch=16,
        workers=1,
    )