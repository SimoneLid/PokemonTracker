py -3.10 yolov5/train.py ^
--data models/stream.yaml ^
--weights yolov5\runs\train\exp82\weights\best.pt ^
--epoch 100 ^
--img 640 ^
--workers 6 ^
--batch-size 32 ^
--hyp hyp.no-augmentation.yaml ^
--rect ^
--patience 0 ^
--save-period 10 

py -3.10 validation_histogram.py