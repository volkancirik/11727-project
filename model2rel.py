from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, Dense, Flatten, Layer
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop

import theano
from keras.models import model_from_json

import json, time, datetime, os, sys
import cPickle as pickle
import numpy as np
import math
from prepare_data import vectorize_positive
from ranking import Ranking
BAN_LIST = set(['/m/08mbj32'])
'''
test a MMMT model
'''
def load_model(path):
	meta = path + '.meta'
	arch = path + '.arch'
	model_filename = path + '.model'

	meta_dict = pickle.load(open(meta))

	with open(arch) as json_file:
		architecture = json.load(json_file)
	model = model_from_json(architecture)
	model.load_weights(model_filename)

	return model, meta_dict

def print_rep(vec):
	new_vec = [str(math.ceil((val*1000))/1000.0) for val in vec]
	print " ".join(new_vec)

def vectorize_all(descriptions, max_len = 30):

	N = len(descriptions)
	X_all = np.zeros((N,max_len), dtype = 'int64')

	mid2idx = {}
	for i,mid in enumerate(descriptions):
		v = descriptions[mid]['idx_vector']
		X_all[i,max_len - len(v):] = v
		mid2idx[mid] = i
	return X_all, mid2idx

model,meta_dict = load_model(sys.argv[1])
w = model.nodes['e_relation'].get_weights()

for i,rel in enumerate(meta_dict['dicts']['rel_idx']):
	vec = " ".join([str(v) for v in w[0][i]])
	print "\t".join([rel.split('/')[1]+'/'+rel.split('/')[-1],vec]).encode('utf-8')
