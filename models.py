from keras.models import Model, Sequential, load_model
from keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D, RepeatVector, Reshape
from keras.layers.merge import Concatenate, Multiply, Dot, Subtract
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
# from resnet50 import ResNet50

from keras import backend as K
K.set_image_dim_ordering('th')

def ObjectNet(object_dim):

	object_input = Input(shape = (3, object_dim, object_dim), name = 'object_input')

	x = Conv2D(64, 3, activation='relu', padding = 'same')(object_input)
	x = Conv2D(64, 3, activation='relu', padding = 'same')(x)
	x = MaxPooling2D(pool_size = 2)(x)

	# Block 2
	x = Conv2D(128, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(128, 3, activation='relu', padding = 'same')(x)
	x = MaxPooling2D(pool_size = 2)(x)

	# Block 3
	x = Conv2D(256, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(256, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(256, 3, activation='relu', padding = 'same')(x)
	x = MaxPooling2D(pool_size = 2)(x)

	# Block 4
	x = Conv2D(512, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same',)(x)
	x = MaxPooling2D(pool_size = 2)(x)

	# Block 5
	x = Conv2D(512, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same')(x)
	x = MaxPooling2D(pool_size = 2)(x)

	x = Flatten()(x)
	x = Dense(4096, activation = 'relu')(x)
	x = Dense(1024, activation = 'relu', name = 'object_vector')(x)
	output = Dense(80, activation = 'softmax', name = 'objectnet_output')(x)

	model = Model(inputs = object_input, outputs = output)

	return model

def AddGist(objectnet, context_dim):

	k = 5
	context_input = Input(shape = (3, context_dim, context_dim), name = 'context_input')
	x = Conv2D(64, k, activation='relu', strides = 2, padding='valid')(context_input)
	x = Conv2D(64, k, activation='relu', strides = 2, padding='valid')(x)
	x = Conv2D(128, k, activation='relu', strides = 2, padding='valid')(x)
	x = Conv2D(128, k, activation='relu', strides = 2, padding='valid')(x)
	x = Conv2D(256, k, activation='relu', strides = 2, padding='valid')(x)
	x = Conv2D(256, 3, activation='relu', strides = 2, padding='valid')(x)
	x = Conv2D(512, 3, activation='relu', padding='valid')(x)
	x = Conv2D(512, 3, activation='relu', padding='valid')(x)

	x = Flatten(name = 'gist_vector')(x)
	x = Concatenate()([x, objectnet.get_layer('object_vector').output])
	output = Dense(80, activation = 'softmax', name = 'gistnet_output')(x)

	model = Model(inputs = [objectnet.input, context_input], outputs = output)
	return model

def build_model(add, object_dim, **kwargs):

	if add == 'gist': return AddGist(ObjectNet(object_dim), kwargs['context_dim'])
	else: raise ValueError('Please specify a valid model.')