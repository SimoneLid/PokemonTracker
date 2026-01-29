from ultralytics import YOLO


model = YOLO("yolov8n.pt") 

results = model.train(
    data="pokemon.yaml",
    epochs=50,
    imgsz=640,
    plots=True
)