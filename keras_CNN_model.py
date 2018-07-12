from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Input, Flatten, Activation, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model

def model(input_shape):

	X_input = Input(input_shape)

	X = ZeroPadding2D((3, 3))(X_input)

	X = Conv2D(16, (5, 5), strides = (1, 1))(X)
	X = BatchNormalization(axis = 3)(X)
	X = Activation('relu')(X)

	X = MaxPooling2D((2, 2))(X)
	X = Flatten()(X)
	X = Dense(6, activation = 'softmax')(X)

	model = Model(inputs = X_input, outputs = X)

	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

	return model
	


	