# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import json, time, datetime, os
import cPickle as pickle

from prepare_data import prepare_train, vectorize_negative
from get_model import get_model
from utils import get_parser

parser = get_parser()
p = parser.parse_args()

PATIENCE=10
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
MODEL = p.model
HIDDEN_SIZE = p.n_hidden
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'M' + str(MODEL) + '_H' + str(HIDDEN_SIZE) + '_E' + str(EPOCH)

#X_tr_pos, X_val_pos, X_test_pos, descriptions, dicts, short_descs, triple_set = prepare_train(mode = 'debug')
X_tr_pos, X_val_pos, X_test_pos, descriptions, dicts, short_descs, triple_set = prepare_train()

MAXLEN = X_tr_pos[0].shape[1]
ENT_EMB = len(dicts['word_idx'])
REL_EMB = len(dicts['rel_idx'])

N = X_tr_pos[0].shape[0]
N_val = X_val_pos[0].shape[0]


model = get_model(MODEL, ENT_EMB, REL_EMB, MAXLEN, HIDDEN_SIZE)

train_history = {'loss' : [], 'val_loss' : []}
pickle.dump({'dicts' : dicts , 'short_descs' : short_descs, 'triple_set' : triple_set, 'descriptions' : descriptions },open(PREFIX + FOOTPRINT + '.meta', 'w'))
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)


PIECES = 100
best_val_loss = float('inf')
pat = 0
for iteration in xrange(EPOCH):
	print('_' * 50)

	psize_pos = N / PIECES
	psize_neg = 2 * psize_pos
	X_tr_neg = vectorize_negative('../data/15K/train.txt.filtered', dicts['rel_idx'], descriptions, X_tr_pos[0].shape[1], short_descs, triple_set, negative_samples = 1, verbose = False)
	epoch_loss = 0

	for piece in xrange(PIECES):
		print('iteration {} piece {}/{}'.format(iteration,piece+1,PIECES))
		eh_pos = model.fit({'input1' : X_tr_pos[0][piece*psize_pos : (piece+1)*psize_pos], 'input2' : X_tr_pos[1][piece*psize_pos : (piece+1)*psize_pos], 'input3' : X_tr_pos[2][piece*psize_pos : (piece+1)*psize_pos].reshape((psize_pos,1)), 'output' : np.ones((psize_pos)) }, batch_size = BATCH_SIZE, nb_epoch=1, verbose = True)
		eh_neg = model.fit({'input1' : X_tr_neg[0][piece*psize_neg : (piece+1)*psize_neg], 'input2' : X_tr_neg[1][piece*psize_neg : (piece+1)*psize_neg], 'input3' : X_tr_neg[2][piece*psize_neg : (piece+1)*psize_neg].reshape(psize_neg,1), 'output' : np.zeros((psize_neg)) }, batch_size = BATCH_SIZE, nb_epoch=1, verbose = True)
		epoch_loss = epoch_loss + eh_pos.history['loss'][0] + eh_neg.history['loss'][0]

	val_loss = model.evaluate({'input1' : X_val_pos[0], 'input2' : X_val_pos[1], 'input3' : X_val_pos[2].reshape(N_val,1), 'output' : np.ones((N_val)) }, batch_size = BATCH_SIZE, verbose = True)
	train_history['loss'] += [epoch_loss]
	train_history['val_loss'] += [val_loss]

	print("TR  {} VAL {} best VL {} no improvement in {}".format(train_history['loss'][-1],train_history['val_loss'][-1],best_val_loss,pat))

	if train_history['val_loss'][-1] >= best_val_loss:
	 	pat += 1
	else:
		pat = 0
		best_val_loss = train_history['val_loss'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
		pickle.dump({ 'train_history' : train_history},open(PREFIX + FOOTPRINT + '.log', 'w'))
	if pat == PATIENCE:
		break


