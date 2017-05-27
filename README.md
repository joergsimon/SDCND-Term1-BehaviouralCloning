# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains my realisation of the Behavioral Cloning Project. ConvNets are used here to learn End to End conversions from a camera image mounted on a car to steering angles to drive that car along the road.

The repo has the following file structure:
* [model.py](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/model.py) (script used to create and train the model)
* [drive.py](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/drive.py) (script to drive the car - not modified)
* [model.h5](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/model.h5) (a trained Keras model)
* [report.md](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/report.md) the report writeup file
* [video.mp4](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/video.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)
* [helper](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/tree/master/helper) (python module with functionality like generator, io...)
* [models](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/tree/master/models) (collection of the tried out models. In the end model5 was used)

### `model.py`

`model.py` takes a number as a first input. This number should identify the model in the `models/` directory, and is then used to train the model. It always saves the result in the folder `model-result/` in the form of `model-d{modelnumber}-{epoch_checkpoint}.h5` and the last result in the form `model-d{modelnumber}.h5`

F.e.
```sh
model.py 5 
```
will train the `models/model5.py` model and produce `model-result/model-d5-00.h5` and ongoing models.

### `drive.py` and `video.py`

drive and video are kept as they are provided from udacity. For the case that someone clones this repo, the basic usage description is also kept.

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

`video.py` can use the directory to procude a video:
```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```