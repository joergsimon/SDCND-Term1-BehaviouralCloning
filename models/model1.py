def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model