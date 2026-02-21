python yolov5/train.py --epochs 50 --batch-size -1 --save-period 10 --data models/stream.yaml ^
--weights models/stream_base.pt --workers 0 --device 0 --name stream --img 640 --cfg yolov5/models/yolov5s.yaml ^
--hyp yolov5/data/hyps/hyp.stream.yaml --cos-lr