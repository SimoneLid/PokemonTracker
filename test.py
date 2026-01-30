import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

video_path = "video_pokemon_2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, conf=0.2)

        annotated_frame = results[0].plot()

        cv2.imshow("Pokemon Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()