import matplotlib
import matplotlib.pyplot as plt

def print_hist(history_object):
    print(history_object.history.keys())
    print("loss")
    print(history_object.history['loss'])
    print("val loss")
    print(history_object.history['val_loss'])

def plot(history_object):
    ## plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    return plt