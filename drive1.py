import csv
import cv2
import numpy as np

DATA_PATH = './data/'
IMG_DATA_PATH = DATA_PATH + 'IMG/'
IDX_CENTER_IMG = 0
IDX_STEER_ANGLE = 3

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
        # we have a header, skip that
        continue
    filename = source_path.split('/')[-1]
    current_path = IMG_DATA_PATH + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[IDX_STEER_ANGLE])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
print("x shape", X_train.shape)
print("y shape", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')