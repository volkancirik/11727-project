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
validation = '../data/15K/valid.txt.filtered'

e_all, mid2idx = vectorize_all(meta_dict['descriptions'])

getMerged = theano.function([model.inputs['input1'].input, model.inputs['input2'].input, model.inputs['input3'].input ], model.nodes['merge_siam'].get_output(train = False), allow_input_downcast=True)

getFinal = theano.function([model.inputs['input1'].input, model.inputs['input2'].input, model.inputs['input3'].input ], model.nodes['sigmoid'].get_output(train = False), allow_input_downcast=True)

X_all = getMerged(e_all,e_all,np.ones((e_all.shape[0],1)))

REL_EMB = len(meta_dict['dicts']['rel_idx'])
MERGED_SIZE = X_all.shape[1]
HIDDEN_SIZE = int(sys.argv[2])

cloned_model = Graph()
cloned_model.add_input(name =  'intermediate', input_shape = (MERGED_SIZE,))
cloned_model.add_input(name =  'input3', input_shape = (1,), dtype = 'int')
cloned_model.add_node(Embedding(REL_EMB,HIDDEN_SIZE, input_length = 1), name = 'e_relation', input = 'input3')
cloned_model.add_node(Flatten(), name = 'relation', input = 'e_relation')
cloned_model.add_node(Dense(HIDDEN_SIZE), name = 'final', inputs = ['intermediate','relation'], merge_mode = 'concat', concat_axis = -1)
cloned_model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = 'final')
cloned_model.add_output(name = 'output', input = 'sigmoid')

for node_name in ['e_relation','final','sigmoid']:
	cloned_model.nodes[node_name].set_weights(model.nodes[node_name].get_weights())

getFinalClone = theano.function([cloned_model.inputs['intermediate'].input, cloned_model.inputs['input3'].input ], cloned_model.nodes['sigmoid'].get_output(train = False), allow_input_downcast=True)

llist = []
rlist = []
BATCH_SIZE = 400
lines = [line.strip().split() for line in open(validation)]
N_BATCH = int(X_all.shape[0] / 512) + 1
for instance, line in enumerate(lines):
	[e1,r,e2] = line
	if e1 in BAN_LIST or e2 in BAN_LIST:
		continue
	idx1 = mid2idx[e1]
	idx2 = mid2idx[e2]
	X_repeat = np.repeat(X_all[idx2,:MERGED_SIZE/2].reshape((1,MERGED_SIZE/2)),X_all.shape[0], axis = 0)
	X_left = np.concatenate((X_all[:,:MERGED_SIZE/2], X_repeat), axis = 1)

	X_repeat = np.repeat(X_all[idx1,:MERGED_SIZE/2].reshape((1,MERGED_SIZE/2)),X_all.shape[0], axis = 0)
	X_right = np.concatenate((X_repeat, X_all[:,:MERGED_SIZE/2]), axis = 1)

	X_rel = np.ones((X_all.shape[0],1))
	X_rel.fill(meta_dict['dicts']['rel_idx'][r])


	b = np.random.choice(range(N_BATCH))
	scores_left = getFinalClone(X_left[b*BATCH_SIZE: (b+1)*BATCH_SIZE],X_rel[b*BATCH_SIZE: (b+1)*BATCH_SIZE])
	scores_right = getFinalClone(X_right[b*BATCH_SIZE: (b+1)*BATCH_SIZE],X_rel[b*BATCH_SIZE: (b+1)*BATCH_SIZE])

	score_correct = getFinalClone(np.concatenate((X_all[idx1,:MERGED_SIZE/2].reshape((1,MERGED_SIZE/2)),X_all[idx2,:MERGED_SIZE/2].reshape((1,MERGED_SIZE/2))), axis = 1), np.array(meta_dict['dicts']['rel_idx'][r]).reshape((1,1)) )

	if instance % 10000 == 0:
		print >> sys.stderr, "instance %d/%d " % (instance+1, len(lines))

	l_list = scores_left.reshape((scores_left.shape[0],)).tolist() + score_correct.tolist()[0]
	r_list = scores_right.reshape((scores_right.shape[0],)).tolist() + score_correct.tolist()[0]

	l_set = set(l_list)
	r_set = set(r_list)
	r_left = Ranking(sorted(l_list,reverse = True))
	r_right = Ranking(sorted(r_list,reverse = True))
	left_rank = r_left.rank(score_correct[0][0])
	right_rank = r_right.rank(score_correct[0][0])
	llist += [left_rank]
	rlist += [right_rank]
print sys.argv[1],':',np.mean(llist), np.mean(rlist)

