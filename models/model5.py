from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation, BatchNormalization

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    
    model.add(Conv2D(16, (5, 5), strides=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (5, 5), strides=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (5, 5), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model