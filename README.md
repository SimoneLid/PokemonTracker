# PokemonTracker

This repo contains an object detection model trained to recognize Pokemon in the wild.

### Introduction
The model is trained using a custom version of yolov5 modified to follow the work described in this [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Real-Time_Object_Detection_for_Streaming_Perception_CVPR_2022_paper.pdf). The model tries to predict the motion of the Pokemon in the next frame based on the previous two frames.

### Training
The model was trained in two steps, using the frames in `dataset_stream`:
1. At first we created a model that could recognize still Pokemon, using the current frame and the previous one as support. The weights of this model are in the `stream_base.pt` file.
2. Using the previous model another model was trained, this time using two consecutive frames as input and asking the model to predict the successive one. The weights of the final model are in the `TBD` file.

### Usage
To run a model on a chosen video run:

```bash
python run_model.py <video_path> <weights_path>
```