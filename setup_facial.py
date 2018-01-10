import numpy as np
import os
import re
import scipy.misc
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras import regularizers
from keras.optimizers import SGD
#from ResNet50_noBN import ResNet50_noBN

class FACIAL:
	def __init__(self, resize=None):
		filename = "fer2013/fer2013.csv"
		if not os.path.exists(filename):
			print("Facial data not found: {}".format(filename))
			print("You can download it from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")

		# load data
		count = 0
		train_data = []
		train_labels = []
		validation_data = []
		validation_labels = []
		test_data = []
		test_labels = []
		with open(filename, "r") as f:
			for l in f:
				count += 1
				# skip the first header
				if count == 1:
					continue

				# extract columns
				try:
					label, pixel, usage = l.split(",")
				except ValueError:
					print("End at index index:{}".format(count))
					break

				# use one-hot encoding for labels
				temp_label = int(label)
				label = np.zeros(7)
				label[temp_label] = 1

				# load image
				pixel_value = [int(x) for x in pixel.split()]
				data = np.reshape(pixel_value, (48, 48))
				if resize:
					data = scipy.misc.imresize(data, (resize, resize))
				data = np.expand_dims(data, axis=2)
				data = data/255 - 0.5
				usage = usage.rstrip()
				if usage == "Training":
					train_data.append(data)
					train_labels.append(label)

				elif usage == "PublicTest":
					validation_data.append(data)
					validation_labels.append(label)

				elif usage == "PrivateTest":
					test_data.append(data)
					test_labels.append(label)

				else:
					print("End at index index:{}".format(count))
					break
				
		self.train_data = np.array(train_data)
		self.train_labels = np.array(train_labels)
		self.validation_data = np.array(validation_data)
		self.validation_labels = np.array(validation_labels)
		self.test_data = np.array(test_data)
		self.test_labels = np.array(test_labels)



class FACIALModel:
	def __init__(self, restore = None, session=None, use_log=False, image_size=48):
		self.image_size = image_size
		self.num_channels = 1
		self.num_labels = 7

		if image_size == 48:
			model = resnet.ResnetBuilder.build_resnet_50((self.num_channels, self.image_size, self.image_size), self.num_labels)
		elif image_size == 200:
			input_layer = Input(shape=(self.image_size, self.image_size, 1))
			base_model = ResNet50(weights=None, input_tensor=input_layer)
			x = base_model.output
			x = LeakyReLU()(x)
			x = Dense(128)(x)
			#x = Dropout(0.3)(x)
			x = LeakyReLU()(x)
			#x = Dropout(0.4)(x)
			x = Dense(7)(x)
			if use_log:
				x = Activation("softmax")(x)
			model = Model(inputs=base_model.input, outputs=x)
		else:
			raise Exception("Wrong image size, can only take 48 or 200.")
		

		if restore:
			model.load_weights(restore)

		self.model = model

	def predict(self, data):
		# this is used inside tf session, data should be a tensor
		return self.model(data)
