import json, sys, nltk
import numpy as np
from collections import defaultdict

SHORT_LEN = 20
THRESHOLD = 2
UNK = '<UNK>'
EOS = '</s>'
BAN_LIST = set(['/m/08mbj32'])

def vectorize_negative(fname, rel_idx, descriptions, max_len, short_descs, triple_set, negative_samples = 1, verbose = False):

	lines = [ line.strip().split() for line in open(fname)]
	N = len(lines) * negative_samples * 2
	X_1 = np.zeros((N,max_len), dtype = 'int64')
	X_2 = np.zeros((N,max_len), dtype = 'int64')
	X_3 = np.zeros((N), dtype = 'int64')

	length = []
	count = 0
	for i,line in enumerate(lines):
		[e1, r ,e2 ] = line
		if e1 in BAN_LIST or e2 in BAN_LIST:
			continue

		e = list(np.random.choice( triple_set[e2][r], negative_samples))
		count += negative_samples

		for j in xrange(negative_samples):
			v1 = descriptions[e[j]]['idx_vector']
			v2 = descriptions[e2]['idx_vector']

			X_1[i*negative_samples + j, max_len - len(v1):] = v1
			X_2[i*negative_samples + j, max_len - len(v2):] = v2
			X_3[i] = rel_idx[r]

		try:
			e = list(np.random.choice(triple_set[e1][r], negative_samples))
		except:
			print >> sys.stderr, "triple_set:", triple_set[e1][r]
			print >> sys.stderr, "e1 & r:", e1,r
			print >> sys.stderr, negative_samples
			quit(0)
		count += negative_samples
		for j in xrange(negative_samples):
			v1 = descriptions[e1]['idx_vector']
			v2 = descriptions[e[j]]['idx_vector']

			X_1[i*(negative_samples + 1) + j, max_len - len(v1):] = v1
			X_2[i*(negative_samples + 1) + j, max_len - len(v2):] = v2
			X_3[i] = rel_idx[r]
		if verbose and count % 1000 == 0:
			print count, N
	return X_1, X_2, X_3

def vectorize_positive(fname, rel_idx, descriptions, max_len):

	print "reading file %s" % (fname)
	lines = [ line.strip().split() for line in open(fname)]
	print "file %s is read" % (fname)
	N = len(lines)
	X_1 = np.zeros((N,max_len), dtype = 'int64')
	X_2 = np.zeros((N,max_len), dtype = 'int64')
	X_3 = np.zeros((N), dtype = 'int64')

	length = []
	for i,line in enumerate(lines):
		[e1, r ,e2 ] = line
		if e1 in BAN_LIST or e2 in BAN_LIST:
			continue

		v1 = descriptions[e1]['idx_vector']
		v2 = descriptions[e2]['idx_vector']

		X_1[i,max_len - len(v1):] = v1
		X_2[i,max_len - len(v2):] = v2
		X_3[i] = rel_idx[r]

	return X_1, X_2, X_3

def prepare_train( prefix = '../data/15K/', desc = 'description+missing.json', train = 'train.txt.filtered', valid = 'valid.txt.filtered', test = 'test.txt.filtered', mode = '', clip = 30):

	if mode == 'debug':
		train = valid
		test = valid

	with open(prefix + desc) as data_file:
		desc_list = json.load(data_file)

	vocab = defaultdict(int)
	for e in desc_list:
		if e['mid'] in BAN_LIST:
			continue
		d = nltk.word_tokenize(e['description'].lower().strip())
		for token in d:
			vocab[token] += 1

	### remove non-frequent words
	words = [w for w in vocab]
	for w in words:
		if vocab[w] < THRESHOLD:
			del vocab[w]

	vocab[UNK] = THRESHOLD
	vocab[EOS] = THRESHOLD
	word_idx = dict((c, i) for i, c in enumerate(vocab))
	idx_word = dict((i, c) for i, c in enumerate(vocab))
	V = len(word_idx)
	first_w = idx_word[0]
	idx_word[0] = '*dummy*'
	idx_word[V] = first_w
	word_idx[first_w] = V
	word_idx['*dummy*'] = 0

	print >> sys.stderr, "dictionaries has been created with vocab size {}".format(len(word_idx))

	descriptions = defaultdict(dict)
	max_len = 0
	min_len = 1e6
	short_descs = []
	for e in desc_list:
		if e['mid'] in BAN_LIST:
			continue
		d = nltk.word_tokenize(e['description'].lower().strip()) + [EOS]
		d = d[:clip]
		idx_vector = [ word_idx.get(w, word_idx[UNK]) for w in d]

		descriptions[e['mid']]['idx_vector'] = idx_vector
		descriptions[e['mid']]['description'] = " ".join(d)
		descriptions[e['mid']]['name'] = e['name']
		max_len = max_len if max_len > len(idx_vector) else len(idx_vector)
		min_len = min_len if min_len < len(idx_vector) else len(idx_vector)
		if len(idx_vector) <= SHORT_LEN:
			short_descs += [e['mid']]
	print >> sys.stderr, "min/max description length {}/{} and {} short negative samples".format(min_len,max_len,len(short_descs))

	vocab = defaultdict(int)
	triple_set = {}

	for line in open(prefix + train):
		[e1, r , e2] = line.strip().split()
		if e1 in BAN_LIST or e2 in BAN_LIST:
			continue
		if e1 not in descriptions or e2 not in descriptions:
			print >> sys.stderr, "{} or {} not in descriptions file".format(e1,e2)
			quit(1)
		if e1 not in triple_set:
			triple_set[e1] = defaultdict(set)
		if e2 not in triple_set:
			triple_set[e2] = defaultdict(set)
		triple_set[e1][r].add(e2)
		triple_set[e2][r].add(e1)
		vocab[r] += 1

	for e in triple_set:
		for r in triple_set[e]:
			triple_set[e][r] = list(set(short_descs).difference(triple_set[e]))

	rel_idx = dict((c, i) for i, c in enumerate(vocab))
	idx_rel = dict((i, c) for i, c in enumerate(vocab))

	print >> sys.stderr, "{} unique relations".format(len(rel_idx))

	X_tr_pos = vectorize_positive(prefix+train, rel_idx, descriptions, max_len)
	X_val_pos = vectorize_positive(prefix+valid, rel_idx, descriptions, max_len)
	X_test_pos = vectorize_positive(prefix+test, rel_idx, descriptions, max_len)

	# negative_samples = 1
	# X_tr_neg = vectorize_negative(prefix+train, rel_idx, descriptions, X_tr_pos[0].shape[1], short_descs, triple_set, negative_samples = negative_samples)
	# N = X_tr_pos[0].shape[0]* ( negative_samples * 2 + 1 )
	# X_1 = np.zeros((N,max_len), dtype = 'int64')
	# X_2 = np.zeros((N,max_len), dtype = 'int64')
	# X_3 = np.zeros((N), dtype = 'int64')

	# print >> sys.stderr, "merging negative & positive instances {} {} {}".format(N, X_tr_pos[0].shape,X_tr_neg[0].shape)
	# for i in xrange(X_tr_pos[0].shape[0]):
	# 	X_1[ i*(negative_samples*2 +1) ] = X_tr_pos[0][i]
	# 	X_2[ i*(negative_samples*2 +1) ] = X_tr_pos[1][i]
	# 	X_3[ i*(negative_samples*2 +1) ] = X_tr_pos[2][i]
	# 	for j in xrange( negative_samples*2):
	# 		X_1[ i*(negative_samples*2) + 1 + j] = X_tr_neg[0][ i*(negative_samples*2) + j]
	# 		X_2[ i*(negative_samples*2) + 1 + j] = X_tr_neg[1][ i*(negative_samples*2) + j]
	# 		X_3[ i*(negative_samples*2) + 1 + j] = X_tr_neg[2][ i*(negative_samples*2) + j]
	# print >> sys.stderr, "merging completed."

	dicts = {'idx_word' : idx_word, 'word_idx' : word_idx, 'rel_idx' : rel_idx, 'idx_rel' : idx_rel}
	return X_tr_pos, X_val_pos, X_test_pos, descriptions, dicts, short_descs, triple_set

if __name__ == '__main__':
	X_tr_pos, X_val_pos, X_test_pos, descriptions, dicts, short_descs, triple_set = prepare_train()
	print X_tr_pos[0].shape, X_tr_pos[1].shape, X_tr_pos[2].shape
	print X_val_pos[0].shape
	print X_test_pos[0].shape

 	X_tr_neg = vectorize_negative('../data/15K/train.txt.filtered', dicts['rel_idx'], descriptions, X_tr_pos[0].shape[1], short_descs, triple_set)
	print X_tr_neg[0].shape, X_tr_neg[1].shape, X_tr_neg[2].shape
