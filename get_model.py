from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, Dense, Flatten, Layer, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.optimizers import RMSprop
import keras.backend.tensorflow_backend as K
import os
import tensorflow as tf

def model0(model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE, nb_filter = 256, filter_length = 3):

	model.add_shared_node(Embedding(ENT_EMB,HIDDEN_SIZE, input_length = MAXLEN),name = 'embedding', inputs = ['input1','input2'])
	model.add_shared_node(Convolution1D( nb_filter=nb_filter,filter_length=filter_length, border_mode='valid',activation='relu', subsample_length=1), name='conv1', inputs=['embedding'])
	model.add_shared_node(MaxPooling1D(pool_length=2), name='pool1', inputs=['conv1'])
	model.add_shared_node(Flatten(), name='flatten', inputs=['pool1'])
	model.add_shared_node(Layer(), name='merge_siam', inputs=['flatten'], merge_mode = 'concat', concat_axis = -1)

	return model

def model1(model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE, nb_filter = 250, filter_length = 3):

	RNN = recurrent.GRU
	model.add_shared_node(Embedding(ENT_EMB,HIDDEN_SIZE, input_length = MAXLEN),name = 'embedding', inputs = ['input1','input2'])
	model.add_shared_node(Convolution1D( nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1), name='conv1', inputs=['embedding'])
	model.add_shared_node(MaxPooling1D(pool_length=2), name='pool1', inputs=['conv1'])
	model.add_shared_node(RNN(HIDDEN_SIZE), name='rnn', inputs=['pool1'])
	model.add_shared_node(Layer(), name='merge_siam', inputs=['rnn'], merge_mode = 'concat', concat_axis = -1)

	return model

def model2(model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE):

	RNN = recurrent.GRU

	model.add_shared_node(Embedding(ENT_EMB,HIDDEN_SIZE, input_length = MAXLEN),name = 'embedding', inputs = ['input1','input2'])
	model.add_shared_node(RNN(HIDDEN_SIZE, return_sequences = False), name='rnn1', inputs=['embedding'])
	model.add_shared_node(Dense(HIDDEN_SIZE), name='dense1', inputs=['rnn1'])
	model.add_shared_node(Layer(), name='merge_siam', inputs=['dense1'], merge_mode = 'concat', concat_axis = -1)

	return model

def model3(model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE):

	RNN = recurrent.GRU

	model.add_shared_node(Embedding(ENT_EMB,HIDDEN_SIZE, input_length = MAXLEN),name = 'embedding', inputs = ['input1','input2'])
	model.add_shared_node(RNN(HIDDEN_SIZE), name='rnn1_forward', inputs=['embedding'])
	model.add_shared_node(RNN(HIDDEN_SIZE, go_backwards = True), name='rnn1_backward', inputs=['embedding'])
	model.add_shared_node(Dense(HIDDEN_SIZE), name='dense1', inputs=['rnn1_forward', 'rnn1_backward'])
	model.add_shared_node(Layer(), name='merge_siam', inputs=['dense1'], merge_mode = 'concat', concat_axis = -1)

	return model

def model4(model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE):
	from keras.layers.averagelayer import Average
	from keras.regularizers import l1l2, l2
	model.add_shared_node(Embedding(ENT_EMB,HIDDEN_SIZE, input_length = MAXLEN),name = 'embedding', inputs = ['input1','input2'])
	model.add_shared_node(Average(), name='avg', inputs=['embedding'])
	prev = 'avg'
	for layer in xrange(3):
		model.add_shared_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name='dense'+str(layer+1), inputs=[prev])
		model.add_shared_node(Dropout(0.25),name='dense'+str(layer+1) + '_d', inputs = ['dense'+str(layer+1)] )
		prev = 'dense'+str(layer+1)+'_d'
	model.add_shared_node(Layer(), name='merge_siam', inputs=[prev], merge_mode = 'concat', concat_axis = -1)

	return model


def get_model(m_id, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE,gpu_fraction=0.3):
	models = {0 : model0, 1 : model1, 2 : model2, 3 : model3,  4 : model4}

	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	# with K.tf.device('/gpu:2'):
	# 	K._set_session(K.tf.Session(config=K.tf.ConfigProto(gpu_options = gpu_options)))
	# 	model = Graph()
	# 	model.add_input(name =  'input1', input_shape = (MAXLEN,), dtype = 'int')
	# 	model.add_input(name =  'input2', input_shape = (MAXLEN,), dtype = 'int')
	# 	model.add_input(name =  'input3', input_shape = (1,), dtype = 'int')

	# 	model =  models[m_id](model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE)

	# 	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = 'final')
	# 	model.add_output(name = 'output', input = 'sigmoid')
	# 	optimizer = RMSprop()
	# 	model.compile(loss = { 'output' : 'binary_crossentropy'}, optimizer= optimizer)
	# 	return model

	model = Graph()
	model.add_input(name =  'input1', input_shape = (MAXLEN,), dtype = 'int')
	model.add_input(name =  'input2', input_shape = (MAXLEN,), dtype = 'int')
	model.add_input(name =  'input3', input_shape = (1,), dtype = 'int')

	model =  models[m_id](model, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE)

	model.add_node(Embedding(REL_EMB,HIDDEN_SIZE, input_length = 1), name = 'e_relation', input = 'input3')
	model.add_node(Flatten(), name = 'relation', input = 'e_relation')
	model.add_node(Dense(HIDDEN_SIZE), name = 'final', inputs = ['merge_siam','relation'], merge_mode = 'concat', concat_axis = -1)
	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = 'final')
	model.add_output(name = 'output', input = 'sigmoid')
	optimizer = RMSprop()
	model.compile(loss = { 'output' : 'binary_crossentropy'}, optimizer= optimizer)
	return model
