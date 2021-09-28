# baseline cnn model for mnist
from keras.backend import batch_normalization
from keras.layers.normalization.batch_normalization import BatchNormalization
from numpy import mean
from numpy import std
from numpy import argmax
from keras.models import load_model
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

def load_dataset():
	# load dataset and plot samples
	# output: trains and test sets
	(trainX, trainY), (testX, testY) = mnist.load_data()
    	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
'''
	## to visualize data
    # summarize loaded dataset
	print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
	print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    # plot first few images
	for i in range(9):
	    # define subplot
	    pyplot.subplot(330 + 1 + i)
	    # plot raw pixel data
	    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
    # show the figure
	pyplot.show()
'''
#(trainX, trainY), (testX, testY) = load_dataset()


## scale pixels: original_values black_to_white [0,255] -> rescale_to [0,1]
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
#(trainX), (testX) = prep_pixels(trainX, testX)

## define cnn model
# output: model 
# Mdeo: conv_Relu_MaxPool+dense_Relu+dense_SoftMax;SGD;crossEntropy
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))) # convolutional front-end: 32 filters->size(3,3)
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten()) # The filter maps can then be flattened to provide features to the classifier
	# From here is a multi-class classification task
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform')) # interpret the feature
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax')) # output= 10 cause 10 classes
	# compile model
	opt = SGD(lr=0.01, momentum=0.9) # stochastic gradient descent optimizer. learning rate=0.01, momentum=0.9
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) # loss func, monitoring->accuracy
	return model
#model = define_model()

## evaluate a model using k-fold cross-validation (10 epochs, batch=32)
# input: training dataset
# output: list of accuracy scores and training histories and best model
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	i = 0
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
		if acc > scores[i-1]:
			bestModel = model
		i+=1
		#print(trainX.shape)
		#prediccion = model.predict(trainX[1][::][::][::])
		#print(prediccion)
	return scores, histories, bestModel
#(scores, histories, model) = evaluate_model(dataX, dataY, n_folds=5)

## save model to H5 file
#input: keras model, fileName.h5
def saveKerasModel(model,fileName):
	model.save(fileName)

## load model from H5 file
# input: fileName.h5
# output: keras model
def loadKerasModel(fileName):
	model = load_model(fileName)
	return model


## plot diagnostic learning curves
# input: histories from train and evaluate
# output: screen plots
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
#summarize_diagnostics(histories)

## summarize model performance
# input: scores recolected during evaluation
# output: plots: calculated mean and standard deviation
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
#summarize_performance(scores)

## run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories, model = evaluate_model(trainX, trainY)
	saveKerasModel(model, "ANN_mnist_writtenNumberImages")
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
	
## precit data
# input: keras model, data (images shape:Nx4)
def predictKerasModel_modelClass(model, data):
	probabilitiesVecto = model.predict(data)
	print(argmax(probabilitiesVecto, axis=1))

run_test_harness()