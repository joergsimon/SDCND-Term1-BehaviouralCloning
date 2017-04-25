import csv

DATA_PATH = './data/'
IDX_CENTER_IMG = 0
IDX_STEER_ANGLE = 4

lines = []
with open(DATA_PATH+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[IDX_CENTER_IMG]
    filename = source_path.split('/')[-1]
    current_path = DATA_PATH + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[IDX_STEER_ANGLE])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')