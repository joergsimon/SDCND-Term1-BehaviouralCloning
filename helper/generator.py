from sklearn.utils import shuffle
import numpy as np

from .io import process_line

def generator(samples, batch_size=32, add_left_right=True, add_trans=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                process_line(batch_sample, images, angles, add_left_right, add_trans)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)