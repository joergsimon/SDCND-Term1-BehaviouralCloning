import os
import sys
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from importlib import import_module
from keras.callbacks import ModelCheckpoint

import helper.constants as const
from helper.generator import generator
import helper.plot as plt


samples = []
with open(const.DATA_PATH+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        source_path = line[const.IDX_CENTER_IMG]
        if source_path == "center":
            continue
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=const.BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=const.BATCH_SIZE)

# add checkpoints
# taken from: http://machinelearningmastery.com/check-point-deep-learning-models-keras/
# and https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184
model_num = sys.argv[1]
checkpoint_file = './model-result/model-d'+model_num+'-{epoch:02d}.h5'
callbacks = [
ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True),
]
m = import_module("models.model"+str(model_num))
model = m.get_model()
history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/const.BATCH_SIZE), validation_data=validation_generator,validation_steps=(len(validation_samples)/const.BATCH_SIZE), epochs=const.EPOCHS, callbacks=callbacks)
            
plt.print_hist(history_object)
#plot = plt.plot(history_object)
#plot.show()
model.save('./model-result/model-d'+model_num+'.h5')