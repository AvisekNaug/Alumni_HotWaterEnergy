"""This script contains the skeleton Keras models
"""

from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras import backend as K
#import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, Reshape, Dropout, BatchNormalization, Activation, multiply, Lambda
from keras.callbacks import TensorBoard
from keras.regularizers import L1L2
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataprocess.plotutils import pred_v_target_plot, pred_v_target_plot_transferlearning

class lstm_model():
	
	
	def __init__(self, saveloc: str, inputdim: int, outputdim: int = 1, input_timesteps: int = 1, output_timesteps: int = 1,
	batch_size = 32, reg_l1: float = 0.01, reg_l2: float = 0.02, period: int = 12, stateful: bool = False,
	modelerror = 'mse', optimizer = 'adam'):

		self.saveloc = saveloc
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.input_timesteps = input_timesteps
		self.output_timesteps = output_timesteps
		self.batch_size = batch_size
		self.l1, self.l2 = reg_l1, reg_l2
		self.period = period
		self.stateful = stateful
		self.modelerror = modelerror
		self.optimizer = optimizer

		# time gaps in minutes, needed only for human readable results in output file
		self.timegap = self.period*5
		self.epochs = 0

		# possible regularization strategies
		self.regularizers = L1L2(self.l1, self.l2)

		# logging error on each iteration subsequence
		# self.preds_train = []  # each element has (samplesize, outputsequence=1, feature=1)
		# self.preds_test = []  # each element has (samplesize, outputsequence=1, feature=1)

		# create a file to log the error
		if not self.saveloc.endswith('/'):  # attach forward slash if saveloc does not have one
			self.saveloc += '/'
		file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
		file.close()


	# Create the network
	def design_model(self, lstmhiddenlayers: list = [64, 64], densehiddenlayers: list = [], 
	dropoutlist: list = [[],[]], batchnormalizelist : list = [[],[]]):

		self.lstmhiddenlayers = lstmhiddenlayers
		self.densehiddenlayers = densehiddenlayers
		self.dropoutlist = dropoutlist
		self.batchnormalizelist = batchnormalizelist
		 
		# There will be one dense layer to output the targets
		self.densehiddenlayers += [self.outputdim]

		# Checking processors
		if not self.dropoutlist[0]:
			self.dropoutlist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.dropoutlist[0]), "lstmhiddenlayers and dropoutlist[0] must be of same length"

		if not self.dropoutlist[1]:
			self.dropoutlist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.dropoutlist[1]), "densehiddenlayers and dropoutlist[1] must be of same length"
		if not self.batchnormalizelist[0]:
			self.batchnormalizelist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.batchnormalizelist[0]), "lstmhiddenlayers and batchnormalizelist[0] must be of same length"

		if not self.batchnormalizelist[1]:
			self.batchnormalizelist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.batchnormalizelist[1]), "lstmhiddenlayers and batchnormalizelist[1] must be of same length"
		
		#K.clear_session()
		#config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 6})
		#sess = tf.Session(config=config)
		#K.set_session(sess)
		
		# Design the network
		self.input_layer = Input(batch_shape=(None, self.input_timesteps, self.inputdim), name='input_layer')
		self.reshape_layer = Reshape((self.input_timesteps*self.inputdim,),name='reshape_layer')(self.input_layer)
		self.num_op = self.output_timesteps
		self.input = RepeatVector(self.num_op, name='input_repeater')(self.reshape_layer)
		self.out = self.input

		# LSTM layers
		for no_units, dropout, normalize in zip(lstmhiddenlayers, dropoutlist[0], batchnormalizelist[0]):

			self.out = LSTM(no_units, return_sequences=True, stateful = self.stateful)(self.out)  # recurrent_regularizer=self.regularizers,

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		# Dense layers
		activationlist = ['linear']*(len(densehiddenlayers)-1) + ['linear']  # relu activation for all dense layers exept last
		for no_units, dropout, normalize, activation in zip(densehiddenlayers, dropoutlist[1], batchnormalizelist[1], activationlist):

			self.out = Dense(no_units, activation=activation)(self.out)

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		# compile model
		self.model = Model(inputs=self.input_layer, outputs=self.out)
		self.model.compile(loss=self.modelerror, optimizer=self.optimizer)


	def show_model(self,):
		print(self.model.summary())

	def model_callbacks(self,):

		self.modelchkpoint = ModelCheckpoint(self.saveloc+'LSTM_model_{epoch:02d}_{val_loss:.2f}',
		 monitor = 'val_loss', save_best_only = True, period=2)

		self.earlystopping = EarlyStopping(monitor = 'val_loss', patience=8, restore_best_weights=True)

		self.reduclronplateau = ReduceLROnPlateau(monitor = 'val_loss', patience=2, cooldown = 3)

		self.tbCallBack = TensorBoard(log_dir=self.saveloc+'loginfo', batch_size=self.batch_size, histogram_freq=1,
		 write_graph=False, write_images=False, write_grads=True)

	
	def train_model(self, X_train, y_train, X_val, y_val, epochs: int = 100, saveModel: bool = False, initial_epoch = 0):

		# Number of epochs to run
		self.epochs = epochs

		# train the model
		self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, \
			validation_data=(X_val, y_val) , verbose=2, shuffle=False,initial_epoch = initial_epoch, callbacks=[self.modelchkpoint, \
				self.earlystopping, self.reduclronplateau, self.tbCallBack])

		if saveModel:
			self.save_model()

		return self.history


	def save_model(self,):

			self.model.save(self.saveloc+'LSTM_model_{:02d}epochs.hdf5'.format(self.epochs))

	def evaluate_model(self, X_train, y_train, X_test, y_test, y_sc, scaling: bool = True, saveplot: bool = False, Week: int = 0,
	 lag: int = -1, outputdim_names = ['TotalEnergy']):
	

		# evaluate model on data. output -> (nsamples, output_timesteps, outputdim)
		self.preds_train = self.model.predict(X_train, batch_size=self.batch_size)
		self.preds_test = self.model.predict(X_test, batch_size=self.batch_size)

		for i in range(self.outputdim):
			for j in range(self.output_timesteps):

				# log error on training data
				rmse = sqrt(mean_squared_error(self.preds_train[:, j, i], y_train[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_train[:, j, i]))
				mae = mean_absolute_error(self.preds_train[:, j, i], y_train[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('Week No:{}-Time Step {}: Train RMSE={} |Train CVRMSE={} |Train MAE={}\n'.format(Week,j+1, rmse, cvrmse, mae))
				file.close()

				# log error on test data
				rmse = sqrt(mean_squared_error(self.preds_test[:, j, i], y_test[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_test[:, j, i]))
				mae = mean_absolute_error(self.preds_test[:, j, i], y_test[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('Week No:{}-Time Step {}: Test RMSE={} |Test CVRMSE={} |Test MAE={}\n'.format(Week,j+1, rmse, cvrmse, mae))
				file.close()

		if saveplot:

			pred_v_target_plot(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_train, y_train, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="train",Week=Week)

			pred_v_target_plot(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_test, y_test, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="test",Week=Week)

		return [self.preds_train, self.preds_test]

class lstm_model_transferlearning():
	
	
	def __init__(self, saveloc: str, inputdim: int, outputdim: int = 1, input_timesteps: int = 1, output_timesteps: int = 1,
	batch_size = 32, reg_l1: float = 0.01, reg_l2: float = 0.02, period: int = 12, stateful: bool = False,
	modelerror = 'mse', optimizer = 'adam'):

		self.saveloc = saveloc
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.input_timesteps = input_timesteps
		self.output_timesteps = output_timesteps
		self.batch_size = batch_size
		self.l1, self.l2 = reg_l1, reg_l2
		self.period = period
		self.stateful = stateful
		self.modelerror = modelerror
		self.optimizer = optimizer

		# time gaps in minutes, needed only for human readable results in output file
		self.timegap = self.period*5
		self.epochs = 0

		# possible regularization strategies
		self.regularizers = L1L2(self.l1, self.l2)

		# logging error on each iteration subsequence
		# self.preds_train = []  # each element has (samplesize, outputsequence=1, feature=1)
		# self.preds_test = []  # each element has (samplesize, outputsequence=1, feature=1)

		# create a file to log the error
		if not self.saveloc.endswith('/'):  # attach forward slash if saveloc does not have one
			self.saveloc += '/'
		file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
		file.close()


	# Create the network
	def design_network(self, lstmhiddenlayers: list = [64, 64], densehiddenlayers: list = [], 
	dropoutlist: list = [[],[]], batchnormalizelist : list = [[],[]]):

		self.lstmhiddenlayers = lstmhiddenlayers
		self.densehiddenlayers = densehiddenlayers
		self.dropoutlist = dropoutlist
		self.batchnormalizelist = batchnormalizelist
		 
		# There will be one dense layer to output the targets
		self.densehiddenlayers += [self.outputdim]

		# Checking processors
		if not self.dropoutlist[0]:
			self.dropoutlist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.dropoutlist[0]), "lstmhiddenlayers and dropoutlist[0] must be of same length"

		if not self.dropoutlist[1]:
			self.dropoutlist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.dropoutlist[1]), "densehiddenlayers and dropoutlist[1] must be of same length"
		if not self.batchnormalizelist[0]:
			self.batchnormalizelist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.batchnormalizelist[0]), "lstmhiddenlayers and batchnormalizelist[0] must be of same length"

		if not self.batchnormalizelist[1]:
			self.batchnormalizelist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.batchnormalizelist[1]), "lstmhiddenlayers and batchnormalizelist[1] must be of same length"
		
		
		# Design the network
		self.input_layer = Input(batch_shape=(None, self.input_timesteps, self.inputdim), name='input_layer')
		self.reshape_layer = Reshape((self.input_timesteps*self.inputdim,),name='reshape_layer')(self.input_layer)
		self.num_op = self.output_timesteps
		self.input = RepeatVector(self.num_op, name='input_repeater')(self.reshape_layer)
		self.out = self.input

		# LSTM layers
		for no_units, dropout, normalize in zip(lstmhiddenlayers, dropoutlist[0], batchnormalizelist[0]):

			self.out = LSTM(no_units, return_sequences=True, stateful = self.stateful)(self.out)  # recurrent_regularizer=self.regularizers,

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		# Dense layers
		activationlist = ['linear']*(len(densehiddenlayers)-1) + ['linear']  # relu activation for all dense layers exept last
		for no_units, dropout, normalize, activation in zip(densehiddenlayers, dropoutlist[1], batchnormalizelist[1], activationlist):

			self.out = Dense(no_units, activation=activation)(self.out)

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		self.model = Model(inputs=self.input_layer, outputs=self.out)

	def model_compile(self,):
		# compile model
		self.model.compile(loss=self.modelerror, optimizer=self.optimizer)


	def show_model(self,):
		print(self.model.summary())

	def model_callbacks(self,):

		self.modelchkpoint = ModelCheckpoint(self.saveloc+'LSTM_model_{epoch:02d}_{val_loss:.2f}',
		 monitor = 'val_loss', save_best_only = True, period=2)

		self.earlystopping = EarlyStopping(monitor = 'val_loss', patience=8, restore_best_weights=True)

		self.reduclronplateau = ReduceLROnPlateau(monitor = 'val_loss', patience=2, cooldown = 3)

		#self.tbCallBack = TensorBoard(log_dir=self.saveloc+'loginfo', batch_size=self.batch_size, histogram_freq=1,
		 #write_graph=False, write_images=False, write_grads=True)

	
	def train_model(self, X_train, y_train, X_val, y_val, epochs: int = 100, saveModel: bool = False, initial_epoch = 0):

		# Number of epochs to run
		self.epochs = epochs

		# train the model
		self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, \
			validation_data=(X_val, y_val) , verbose=2, shuffle=False,initial_epoch = initial_epoch, \
				callbacks=[self.modelchkpoint, self.earlystopping,self.reduclronplateau])

		if saveModel:
			self.save_model()

		return self.history


	def save_model(self,):

			self.model.save(self.saveloc+'LSTM_model_{:02d}epochs.hdf5'.format(self.epochs))

	def evaluate_model(self, X_train, y_train, X_test, y_test, y_sc, scaling: bool = True, saveplot: bool = False, Idx: int = 0,
	 lag: int = -1, outputdim_names = ['TotalEnergy']):
	

		# evaluate model on data. output -> (nsamples, output_timesteps, outputdim)
		self.preds_train = self.model.predict(X_train, batch_size=self.batch_size)
		self.preds_test = self.model.predict(X_test, batch_size=self.batch_size)

		for i in range(self.outputdim):
			for j in range(self.output_timesteps):

				# log error on training data
				rmse = sqrt(mean_squared_error(self.preds_train[:, j, i], y_train[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_train[:, j, i]))
				mae = mean_absolute_error(self.preds_train[:, j, i], y_train[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('{}-Time Step {}: Train RMSE={} |Train CVRMSE={} |Train MAE={}\n'.format(Idx,j+1, rmse, cvrmse, mae))
				file.close()

				# log error on test data
				rmse = sqrt(mean_squared_error(self.preds_test[:, j, i], y_test[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_test[:, j, i]))
				mae = mean_absolute_error(self.preds_test[:, j, i], y_test[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('{}-Time Step {}: Test RMSE={} |Test CVRMSE={} |Test MAE={}\n'.format(Idx,j+1, rmse, cvrmse, mae))
				file.close()

		if saveplot:

			pred_v_target_plot_transferlearning(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_train, y_train, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="train",Idx=Idx)

			pred_v_target_plot_transferlearning(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_test, y_test, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="test",Idx=Idx)

		return [self.preds_train, self.preds_test]

class seq2seq_model():
	
	
	def __init__(self, saveloc: str, inputdim: int, outputdim: int = 1, input_timesteps: int = 1, output_timesteps: int = 1,
	batch_size = 32, reg_l1: float = 0.01, reg_l2: float = 0.02, period: int = 12, stateful: bool = False,
	modelerror = 'mse', optimizer = 'adam', hybird_modes = 5):

		self.saveloc = saveloc
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.input_timesteps = input_timesteps
		self.output_timesteps = output_timesteps
		self.batch_size = batch_size
		self.l1, self.l2 = reg_l1, reg_l2
		self.period = period
		self.stateful = stateful
		self.modelerror = modelerror
		self.optimizer = optimizer

		# possible modes
		self.hybird_modes = hybird_modes

		# time gaps in minutes, needed only for human readable results in output file
		self.timegap = self.period*5
		self.epochs = 0

		# possible regularization strategies
		self.regularizers = L1L2(self.l1, self.l2)

		# logging error on each iteration subsequence
		# self.preds_train = []  # each element has (samplesize, outputsequence=1, feature=1)
		# self.preds_test = []  # each element has (samplesize, outputsequence=1, feature=1)

		# create a file to log the error
		if not self.saveloc.endswith('/'):  # attach forward slash if saveloc does not have one
			self.saveloc += '/'
		file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
		file.close()


	# Create the network
	def design_network(self, lstmhiddenlayers: list = [64, 64], densehiddenlayers: list = [], 
	dropoutlist: list = [[],[]], batchnormalizelist : list = [[],[]]):

		self.lstmhiddenlayers = lstmhiddenlayers
		self.densehiddenlayers = densehiddenlayers
		self.dropoutlist = dropoutlist
		self.batchnormalizelist = batchnormalizelist
		 
		# There will be one dense layer with self.hybird_modes units
		self.densehiddenlayers += [self.hybird_modes]

		# Checking processors
		if not self.dropoutlist[0]:
			self.dropoutlist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.dropoutlist[0]), "lstmhiddenlayers and dropoutlist[0] must be of same length"

		if not self.dropoutlist[1]:
			self.dropoutlist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.dropoutlist[1]), "densehiddenlayers and dropoutlist[1] must be of same length"
		if not self.batchnormalizelist[0]:
			self.batchnormalizelist[0] = [False] * (len(self.lstmhiddenlayers))
		else:
			assert len(self.lstmhiddenlayers)==len(self.batchnormalizelist[0]), "lstmhiddenlayers and batchnormalizelist[0] must be of same length"

		if not self.batchnormalizelist[1]:
			self.batchnormalizelist[1] = [False] * (len(self.densehiddenlayers))
		else:
			assert len(self.densehiddenlayers)==len(self.batchnormalizelist[1]), "lstmhiddenlayers and batchnormalizelist[1] must be of same length"
		
		
		# Design the network
		self.input_data = Input(batch_shape=(None, self.input_timesteps, self.inputdim))
		self.input_layer = self.input_data
		return_seqs_list = [True]*(int(len(lstmhiddenlayers)/2 - 1)) + [False]

		# encoding LSTM layers -> latent representation
		for no_units, dropout, normalize, return_seqs in zip(lstmhiddenlayers[0:int(len(lstmhiddenlayers)/2)],
		 										dropoutlist[0][0:int(len(lstmhiddenlayers)/2)],
		  										batchnormalizelist[0][0:int(len(lstmhiddenlayers)/2)],
												return_seqs_list):

			self.input_layer = LSTM(no_units, return_sequences=return_seqs, stateful = self.stateful)(self.input_layer)

			if dropout:
				self.input_layer = Dropout(0.2)(self.input_layer)

			if normalize:
				self.input_layer = BatchNormalization()(self.input_layer)

		# Latent Representation -> Repeat Layers for output_sequence length
		self.num_op = self.output_timesteps
		self.repeat = RepeatVector(self.num_op)(self.input_layer)
		self.input_layer = self.repeat

		# decoding LSTM layers -> output sequence
		for no_units, dropout, normalize in zip(lstmhiddenlayers[int(len(lstmhiddenlayers)/2):],
		 										dropoutlist[0][int(len(lstmhiddenlayers)/2):],
		  										batchnormalizelist[0][int(len(lstmhiddenlayers)/2):]):

			self.input_layer = LSTM(no_units, return_sequences=True, stateful = self.stateful)(self.input_layer)

			if dropout:
				self.input_layer = Dropout(0.2)(self.input_layer)

			if normalize:
				self.input_layer = BatchNormalization()(self.input_layer)

		# select possible modes
		self.mode = Dense(8, activation='relu')(self.input_layer)
		self.mode = Dense(self.hybird_modes, activation='softmax')(self.mode)
		# extract max
		# self.mode = K.argmax(self.mode, axis = -1)

		# Dense layers
		activationlist = ['linear']*(len(densehiddenlayers)-1) + ['linear']  # relu activation for all dense layers exept last
		for no_units, dropout, normalize, activation in zip(densehiddenlayers, dropoutlist[1], batchnormalizelist[1], activationlist):

			self.input_layer = Dense(no_units, activation=activation)(self.input_layer)

			if dropout:
				self.input_layer = Dropout(0.2)(self.input_layer)

			if normalize:
				self.input_layer = BatchNormalization()(self.input_layer)

		# Now select layers with highest probability
		# shape of self.mode (batch_size, output_timesteps, self.modes)
		# shape of self.input_layer = (batch_size, output_timesteps, self.modes)
		# the needed output is (batch_size, output_timesteps, 1)

		#1st approach: do an expectation across last dimension
		self.output = Lambda(self.dotproduct)([self.mode, self.input_layer])

		self.model = Model(inputs=self.input_data, outputs=self.output)

	def dotproduct(self, tensorlist):
		return K.sum(multiply([tensorlist[0], tensorlist[1]]), axis=-1, keepdims=True)

	def model_compile(self,):
		# compile model
		self.model.compile(loss=self.modelerror, optimizer=self.optimizer)


	def show_model(self,):
		print(self.model.summary())

	def model_callbacks(self,):

		self.modelchkpoint = ModelCheckpoint(self.saveloc+'LSTM_model_{epoch:02d}_{val_loss:.2f}',
		 monitor = 'val_loss', save_best_only = True, period=2)

		self.earlystopping = EarlyStopping(monitor = 'val_loss', patience=8, restore_best_weights=True)

		self.reduclronplateau = ReduceLROnPlateau(monitor = 'val_loss', patience=2, cooldown = 3)

		#self.tbCallBack = TensorBoard(log_dir=self.saveloc+'loginfo', batch_size=self.batch_size, histogram_freq=1,
		 #write_graph=False, write_images=False, write_grads=True)

	
	def train_model(self, X_train, y_train, X_val, y_val, epochs: int = 100, saveModel: bool = False, initial_epoch = 0):

		# Number of epochs to run
		self.epochs = epochs

		# train the model
		self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, \
			validation_data=(X_val, y_val) , verbose=2, shuffle=False,initial_epoch = initial_epoch, \
				callbacks=[self.modelchkpoint, self.earlystopping,self.reduclronplateau])

		if saveModel:
			self.save_model()

		return self.history


	def save_model(self,):

			self.model.save(self.saveloc+'LSTM_model_{:02d}epochs.hdf5'.format(self.epochs))

	def evaluate_model(self, X_train, y_train, X_test, y_test, y_sc, scaling: bool = True, saveplot: bool = False, Idx: int = 0,
	 lag: int = -1, outputdim_names = ['TotalEnergy']):
	

		# evaluate model on data. output -> (nsamples, output_timesteps, outputdim)
		self.preds_train = self.model.predict(X_train, batch_size=self.batch_size)
		self.preds_test = self.model.predict(X_test, batch_size=self.batch_size)

		for i in range(self.outputdim):
			for j in range(self.output_timesteps):

				# log error on training data
				rmse = sqrt(mean_squared_error(self.preds_train[:, j, i], y_train[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_train[:, j, i]))
				mae = mean_absolute_error(self.preds_train[:, j, i], y_train[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('{}-Time Step {}: Train RMSE={} |Train CVRMSE={} |Train MAE={}\n'.format(Idx,j+1, rmse, cvrmse, mae))
				file.close()

				# log error on test data
				rmse = sqrt(mean_squared_error(self.preds_test[:, j, i], y_test[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_test[:, j, i]))
				mae = mean_absolute_error(self.preds_test[:, j, i], y_test[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('{}-Time Step {}: Test RMSE={} |Test CVRMSE={} |Test MAE={}\n'.format(Idx,j+1, rmse, cvrmse, mae))
				file.close()

		if saveplot:

			pred_v_target_plot_transferlearning(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_train, y_train, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="train",Idx=Idx)

			pred_v_target_plot_transferlearning(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_test, y_test, self.saveloc, scaling, y_sc, lag = -1, outputdim_names = outputdim_names,
			 typeofplot="test",Idx=Idx)

		return [self.preds_train, self.preds_test]