# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image-center-driving]: ./blob/master/images/center_2017_05_07_18_08_00_409.jpg "Center driving"
[image-recovery1]: ./blob/master/images/center_2017_05_12_14_43_14_831.jpg "Recovery start"
[image-recovery2]: ./blob/master/images/center_2017_05_12_14_43_16_425.jpg "Recovery middle"
[image-recovery3]: ./blob/master/images/center_2017_05_12_14_43_16_983.jpg "Recovery end"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/model.py) (script used to create and train the model)
* [drive.py](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/drive.py) (script to drive the car - not modified)
* [model.h5](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/model.h5) (a trained Keras model)
* [report.md](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/report.md) the report writeup file
* [video.mp4](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/video.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)
* [helper](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/tree/master/helper) (python module with functionality like generator, io...)
* [models](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/tree/master/models) (collection of the tried out models. In the end model5 was used)
* [describe_data.py](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/describe_data.py) (small script do get info about the dat)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file is used to train a specific model. The file however just layouts the general algorithm. Is uses modules in `helper.*` to load the data and transform it inside a generator. It uses `models/model{num}.py` to load specific models and train them.
```sh
python model.py 5
```

Setting of the data directory is not really clean yet, but there is a module constants.py who have a constant for the directory. So if you need to change that, change it there.

### Model Architecture and Training Strategy

Quite a lot of models were explored. The main reason for that was that for a long time no model worked. So I tried data augmentation, collecting data and a lot of models to fix the project. The frustrating thing was, that it were two internal nasty, easy to avoid errors. I mixed up the index over time of the target colum in the training set. So I trained for throttle to predict steering. Fixing that already made a lof of things better. Another error was that I used the off center images for correction. Unfortunately I had an error in that code too, where I in the end loaded the center image again. So I had 3 images of the same but different steering angles. This of course again lead to a really bad performance.

#### 1. An appropriate model architecture has been employed

In the end a relatively small version of the CNN nvidia used for their BB8 car was used. The winning model can be found in [`models/model5.py`](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/models/model5.py). It basically has a set of convolutions with strides, one hidden layer at the end with 100 neurons and a dropout of 0.5. Convolutions used RELU, the hidden layer ELU activations. The idea to use a smaller version comes from several sources: A medium blog post from someone else who did the project explored that you can go to a very tiny network to steer the tracks. Additonally simpler models are faster to train and less prone to overfitting. The final model I used had about 88k trainable parameters. This is still not very small but f.e. way smaller than the original architecture who is approx. at 300k parameters, or other architectors like inception and VGG who goes into the millions.


#### 2. Attempts to reduce overfitting in the model

On the one side the model has a relatively small number of trainable parameters and is not that large. Additonally a Dropout is added one layer befor the final output with a keep probability of 0.5, see [`models/model5.py`](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/models/model5.py) for details. A lot of testing was done, especially since nothing worked because of internal error for about 3-4 weeks.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([`models/model5.py`](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/models/model5.py) line 31).

#### 4. Appropriate training data

I recorded several datasets with a commulative size of 2.5GB with the simulator. Interestingly the current winnind model used 196MB of center lane driving and data agumentation to archive its result. Still I hope to find a good way to employ the other data in the future.

Generally data was recorded with center lane driving. Also I drove the car in the opposite direction. I drove it with recovery moves there I tried to just record only the recovery, and I recorded all that also with the second track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the model from the nvidia paper and then tune it to the different input we have. Since the nvidia model must learn more complex things another idea (inspired by this blog post) was to make the model simpler. I did not go all the path the person in the blog post goes, but at least lowered the amount of trainable parameters from several 100k to about 88k. This network additionally employs a Dropout of 0.5 so the risk of overfitting should be not that high.

Another hint was the progression in training. Earlier more complex models usually started to oscillate between improving and worseing the error on the validation set quite early. This is often seen as a sign of overfitting. The smaller model usually improved in a training for 10 epochs on every step always.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I just drove these spots again, and added different starting positions near them when starting recording.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture ([`models/model5.py`](https://github.com/joergsimon/SDCND-Term1-BehaviouralCloning/blob/master/models/model5.py)) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 60, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 19, 106, 16)       1216      
_________________________________________________________________
batch_normalization_1 (Batch (None, 19, 106, 16)       64        
_________________________________________________________________
activation_1 (Activation)    (None, 19, 106, 16)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 34, 32)         12832     
_________________________________________________________________
batch_normalization_2 (Batch (None, 5, 34, 32)         128       
_________________________________________________________________
activation_2 (Activation)    (None, 5, 34, 32)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 1, 15, 32)         25632     
_________________________________________________________________
batch_normalization_3 (Batch (None, 1, 15, 32)         128       
_________________________________________________________________
activation_3 (Activation)    (None, 1, 15, 32)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 480)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               48100     
_________________________________________________________________
batch_normalization_4 (Batch (None, 100)               400       
_________________________________________________________________
activation_4 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 101       
=================================================================
Total params: 88,601
Trainable params: 88,241
Non-trainable params: 360

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![driving in the center][image-center-driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to always steer back to the middle. I mostly tried to be moderate aggrassive to go back to the middle. These images show what a recovery looks like starting from the right side and then correcting towards the middle:

![start][image-recovery1]
![middle][image-recovery2]
![recovered][image-recovery3]

I also drove the tracks in the other direction. While flipping alread can deal well with the bias towards one direction, I had the feeling that because f.e. of different sight also my driving behaviour was different (especially on track 2), so I guess it was not bad to do that.

From all the recorded data the folloing augmentation was done:
Beside the center image the both side images were taken withough augmentation, but the target value (steering angle) was corrected using a fixed offset. Additionally (again inspired by a blog post) the center image was randomly shiftet along the x axis and the steering angle was adopted to drive agains that shift. Like mentioned in the post this helped for stronger curves and recovery.

All these generated images were then flipped and the angle inverted. This was to counter a bias towards driving into one direction only especially since track 1 would mainly go towards driving left.

The resulting data is fet over a generator to the model. I tried a number of different epoch configurations, but 10 epochs was usually sufficient.
