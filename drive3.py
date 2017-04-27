import csv
import cv2
import numpy as np

DATA_PATH = './data/'
IMG_DATA_PATH = DATA_PATH + 'IMG/'
IDX_CENTER_IMG = 0
IDX_LEFT_IMG = 1
IDX_RIGHT_IMG = 2
IDX_STEER_ANGLE = 4

STEER_CORRECTION_CONSTANT = 0.2

def load_img(csv_line, idx):
    source_path = csv_line[idx]
    filename = source_path.split('/')[-1]
    current_path = IMG_DATA_PATH + filename
    image = cv2.imread(current_path)
    return image

lines = []
with open(DATA_PATH+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[IDX_CENTER_IMG]
    if source_path == "center":
        continue
    
    measurement = float(line[IDX_STEER_ANGLE])
    
    image = load_img(line, IDX_CENTER_IMG)
    images.append(image)
    measurements.append(measurement)
    
    image = load_img(line, IDX_LEFT_IMG)
    images.append(image)
    measurements.append(min(measurement + STEER_CORRECTION_CONSTANT, 1.0))
    
    image = load_img(line, IDX_RIGHT_IMG)
    images.append(image)
    measurement = float(line[IDX_STEER_ANGLE])
    measurements.append(max(measurement - STEER_CORRECTION_CONSTANT, -1.0))

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    augmented_images.append(image_flipped)
    augmented_measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
print("x shape", X_train.shape)
print("y shape", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')