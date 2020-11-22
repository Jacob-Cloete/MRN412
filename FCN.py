%%time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#from __future__ import print_function
from tensorflow import keras
import numpy as np
import pandas as pd
  
nb_epochs = 2000
# = np.array(pd.DataFrame(time_series[:3]))
# = np.array(pd.DataFrame(Class[::3]))

x_train = time_series[::3]
y_train = Class[::3]

x_test = time_series[1::3]
y_test = Class[1::3]

x_real = time_series[2::3]
y_real = Class[2::3]

#test set
nb_classes_real = len(np.unique(y_real))

y_real = (y_real - y_real.min())/(y_real.max()-y_real.min())*(nb_classes_real-1)
Y_real = keras.utils.to_categorical(y_real, nb_classes_real)

real_mean = np.array(x_real).mean()
real_std = x_real.std()
x_real = (x_real - real_mean)/(real_std)
x_real = x_real.reshape(x_real.shape + (1,1,))

#validation set and train set
nb_classes = len(np.unique(y_test))
batch_size = 64

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)


Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)

x_test = (x_test - x_train_mean)/(x_train_std)
x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))

initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)
regularizer = keras.regularizers.l1_l2(l1=0.01, l2=0.01)

x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
conv1 = keras.layers.Conv2D(128, 8, 1, padding='same',kernel_initializer=initializer,
                            kernel_regularizer=regularizer)(x)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

#    drop_out = Dropout(0.2)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, padding='same',kernel_initializer=initializer,
                            kernel_regularizer=regularizer)(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

#    drop_out = Dropout(0.2)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, padding='same',kernel_initializer=initializer,
                            kernel_regularizer=regularizer)(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.GlobalAveragePooling2D()(conv3)
out = keras.layers.Dense(nb_classes, activation='softmax')(full)


model = keras.models.Model(inputs=x, outputs=out)

optimizer = keras.optimizers.Adamax()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.0001) 
hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
#testing set
eva = model.evaluate(x_real,Y_real)

#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
