python yolov5/train.py --epochs 50 --batch-size -1 --save-period 10 --data models/stream.yaml ^
--weights models/single_v4.pt --workers 6 --device 0 --name stream --img 640 --cfg yolov5/models/yolov5s.yaml ^
--hyp yolov5/data/hyps/hyp.stream.yaml --cos-lr