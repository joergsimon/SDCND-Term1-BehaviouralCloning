from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation, BatchNormalization


# model.add(BatchNormalization()) is not used in the first input layer
# and before the output layer. The reson is somehow that for unsupervised
# models this has empirically shown to be bad (see 
# Unsupervised Representation Learning with Deep Convolutional Generative
# Adversarial Networks; Rudford et al. 2016)
# storing the model and calculating the difference with Norm. also at the
# ends would be interesting...

# also NVIDIA already does not use max pooling, but this is also
# recommendet in the DCGAN (== Rudford et al. 2016) paper. Again this
# paper is for something different, but the intution would be it also
# helps here

# ReLU activation is in a Nair & Hinton 2010 paper
# Adam optimizer is in a Kingma & Ba 2014 paper

# "recent studies have shown that there is a direct link between how
# fast models learn and their generalization performance 
# (Hardt et al., 2015)" taken from Rudford et al. 2016
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    
    model.add(Conv2D(24, (5, 5), strides=(2,2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(35, (5, 5), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(48, (5, 5), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3, 3), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(48, (3, 3), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (1, 1), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(3))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model