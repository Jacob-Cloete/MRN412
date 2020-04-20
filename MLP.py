%%time
from __future__ import print_function
 
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(813306)
  
nb_epochs = 2000

#fname = each
x = time_series
y = Class

x_train = x[::3]
y_train = y[::3]
x_test = x[1::3]
y_test = y[1::3]
x_real = x[2::3]
y_real = y[2::3]


#train,validation and testing sets
nb_classes =len(np.unique(y_test))
y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
nb_classes_real =len(np.unique(y_real))
y_real = (y_real - y_real.min())/(y_real.max()-y_real.min())*(nb_classes_real-1)
batch_size = 16

Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)
Y_real = keras.utils.to_categorical(y_real, nb_classes_real)

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
x_test = (x_test - x_train_mean)/(x_train_std)
x_real = (x_real - x_train_mean)/(x_train_std)


x = keras.layers.Input(x_train.shape[1:])
y= keras.layers.Dropout(0.1)(x)
y = keras.layers.Dense(500, activation='relu')(y)
y = keras.layers.Dropout(0.2)(y)
y = keras.layers.Dense(500, activation='relu')(y)
y = keras.layers.Dropout(0.2)(y)
y = keras.layers.Dense(500, activation = 'relu')(y)
y = keras.layers.Dropout(0.3)(y)
out = keras.layers.Dense(nb_classes, activation='softmax')(y)

model = keras.models.Model(inputs=x, outputs=out)

optimizer = keras.optimizers.Adamax()   
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=200, min_lr=0.1)

hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(x_test, Y_test), 
            #callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
             callbacks=[reduce_lr])
model.summary()
#testing set
eva = model.evaluate(x_real,Y_real)

#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
