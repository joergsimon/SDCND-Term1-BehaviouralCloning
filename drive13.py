import os
import csv
import sklearn
import cv2
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation, BatchNormalization
import matplotlib.pyplot as plt

DATA_PATH = './data/'
IMG_DATA_PATH = DATA_PATH + 'IMG/'
IDX_CENTER_IMG = 0
IDX_LEFT_IMG = 1
IDX_RIGHT_IMG = 2
IDX_STEER_ANGLE = 3

BATCH_SIZE = 32
EPOCHS = 7

STEER_CORRECTION_CONSTANT = 0.2

def load_img(csv_line, idx):
    source_path = csv_line[idx]
    filename = source_path.split('/')[-1]
    current_path = IMG_DATA_PATH + filename
    image = cv2.imread(current_path)
    return image
    
def add_image_and_flipped(csv_line, idx, measurment, images, angles):
    image = load_img(line, IDX_CENTER_IMG)
    images.append(image)
    angles.append(measurment)
    
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurment
    images.append(image_flipped)
    angles.append(measurement_flipped)

def process_line(csv_line, images, angles):
    source_path = line[IDX_CENTER_IMG]
    
    measurement = float(line[IDX_STEER_ANGLE])
    
    add_image_and_flipped(csv_line, IDX_CENTER_IMG, 
                            measurement, images, angles)
    
    correct_to_right = max(measurement - STEER_CORRECTION_CONSTANT, -1.0)
    add_image_and_flipped(csv_line, IDX_LEFT_IMG, 
                            correct_to_right, images, angles)
    
    correct_to_left = min(measurement + STEER_CORRECTION_CONSTANT, 1.0)
    add_image_and_flipped(csv_line, IDX_RIGHT_IMG, 
                            correct_to_left, images, angles)
    
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                process_line(batch_sample, images, angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open(DATA_PATH+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        source_path = line[IDX_CENTER_IMG]
        if source_path == "center":
            continue
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(35, (5, 5), strides=(1,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), strides=(2,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(1,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(80, (3, 3), strides=(1,2)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(24, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('elu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.summary()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/BATCH_SIZE), validation_data=validation_generator,validation_steps=(len(validation_samples)/BATCH_SIZE), epochs=EPOCHS)
            
print(history_object.history.keys())
print("loss")
print(history_object.history['loss'])
print("val loss")
print(history_object.history['val_loss'])
### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
#
model.save('model-d13.h5')