# example of creating a CNN with an efficient inception module
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Dense 
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D 
from keras.utils import plot_model

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

#train and validation sets
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

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, 1, padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, 1, padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, 2, padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, 1, padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, 5, padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D(3, strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, 1, padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(200, 1, 1))
# add inception block 1
layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
# add inception block 1
layer = inception_module(layer, 128, 128, 192, 32, 96, 64)

layer = Activation('relu')(layer)
layer = GlobalAveragePooling2D()(layer)
layer = Dense(nb_classes, activation='softmax')(layer)

# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture

%%time
optimizer = keras.optimizers.Adamax()   
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

nb_epoch = 4

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=200, min_lr=0.1)

hist = model.fit(x_train, Y_train, batch_size=32, nb_epoch=nb_epoch,verbose=1, 
                 validation_data=(x_test, Y_test))

print('real data:',model.evaluate(x_real,Y_real))
