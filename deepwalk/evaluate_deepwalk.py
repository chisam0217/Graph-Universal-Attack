#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 

import numpy as np
import sys
import scipy.sparse as sp
import random
import networkx as nx
import argparse
import time
from collections import defaultdict, Counter
from multiprocessing import cpu_count
from gensim.models.word2vec import Vocab
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from utils import load_data, load_polblogs_data
from sklearn.preprocessing import MultiLabelBinarizer
import os.path as op
import graph
from deepwalkmodel import Skipgram
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   COUNT_WORDS                                                                                 #
#    Helper function for Skipgram. Returns dictionary of the times each vertex appear in walks  #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def count_words(walks):
	c = Counter()
	for words in walks:
		c.update(words)
	return c


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   DEEPWALK                                                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def deepwalk(tmp_G, args):
	start_time = time.time()

	# Init graph
	G = graph.Graph(tmp_G)

	# Info about the walks
	total_walks = G.num_of_nodes * args.num_walks
	data_size = total_walks * args.walk_length

	# print("\nNumber of nodes: {}".format(G.num_of_nodes))
	# print("Total number of walks: {}".format(total_walks)) # Number of walks starting from each node
	# print("Data size (walks*length): {}\n".format(data_size))

	# Create the random walks and store them in walks list
	print("Generate walks ...")
	walks = G.build_deep_walks(num_paths=args.num_walks, path_length=args.walk_length)

	# Apply model to each walk = sentence
	print("Applying %s on walks ..."% args.model)
	if args.model == 'skipgram' :
		vertex_counts = count_words(walks) # dictionary of the times each vertex appear in walks
		model = Skipgram(sentences=walks, vocabulary_counts=vertex_counts,size=args.dimension,window=5, min_count=0, trim_rule=None, workers=cpu_count(), iter=args.iter)
	else :
		if args.model == 'word2vec':
			model = Word2Vec(walks, size=args.dimension, window=5, min_count=0, sg=1, hs=1, workers=cpu_count())
		else:
			raise Exception("Unknown model: '%s'.  Valid models: 'word2vec', 'skipgram'" % args.model)

	# Save to output file
	print("----- Total time {:.2f}s -----".format(time.time() - start_time))
	# if cleangraph:
	#   # model.wv.save_word2vec_format(args.output)
	# else:
	#   folder_path = op.join("./" + args.dataset, "perturbation" + str(perturb_idx))
	#   if not op.exists(folder_path):
	#     os.mkdir(folder_path)
	#   file_path = op.join(folder_path, "node" + str(node_idx) + "_" + args.output)

		# model.wv.save_word2vec_format(file_path)
	return model.wv

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Create dictionary (graph) our of sparse matrix                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def sparse2graph(x):
		G = defaultdict(lambda: set())
		cx = x.tocoo()
		for i,j,v in zip(cx.row, cx.col, cx.data):
				G[i].add(j)
		return {str(k): [str(x) for x in v] for k,v in iteritems(G)}


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   HELPER FUNCTIONS FOR SPLITTING DATA TRAIN, VALIDATION, TEST                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def get_splits():
		idx_train = range(200)
		idx_val = range(200, 500)
		idx_test = range(500, 1500)
		return idx_train, idx_val, idx_test

def format_csr(y_):
	y = [[] for x in range(y_.shape[0])]

	cy =  y_.tocoo()
	for i, j in zip(cy.row, cy.col):
			y[i].append(j)
	return y



#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   EVALUATE                                                                                    #
#   Perform Logistic Regression of embedding                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--num-walks', default=10, type=int)
	parser.add_argument('--walk-length', default=40, type=int)
	parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--model', default='word2vec', help='Type of model to apply on walks (word2vec/skipgram)')
	# parser.add_argument("--emb", required=True)
	parser.add_argument('--emb', default='.deepwalk.embeddings', help='the embedding file')
	# parser.add_argument("--net", required=True)
	# parser.add_argument("--labels", required=True)
	parser.add_argument('--dic-network-name', default='network')
	parser.add_argument('--dic-label-name', default='label')
	parser.add_argument('--dataset', default='polblogs', help='dataset')


	args = parser.parse_args()



	#load dataset
	if args.dataset == "polblogs":
	    graph, _, labels, idx_train, idx_test = load_polblogs_data()
	else:

		graph, _, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
	# graph, _, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

	## Load Embeddings
	embeddings_file = op.join(args.dataset, args.dataset + args.emb)
	# model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	model = deepwalk(graph, args)
	# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
	print ('model', model)
	print (len(graph))
	features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])
	
	labels_matrix = np.matrix(labels)
	labels_matrix = sp.csr_matrix(labels_matrix)
	labels = np.reshape(labels, (labels.shape[0], ))

	X, y = features_matrix, labels
	# y = MultiLabelBinarizer().fit_transform(y)
	y_train = y[idx_train]
	y_test = y[idx_test]
	X_train = X[idx_train]
	X_test = X[idx_test]

	# y_train = format_csr(y_train_)
	# y_test = format_csr(y_test_)


	## Logistic Regression

	# Train on data
	logisticRegr = LogisticRegression()
	logisticRegr.fit(X_train, y_train)

	# Measure accuracy
	score = logisticRegr.score(X_test, y_test)
	clean_pred = logisticRegr.predict(X_test)

	# Output results
	print ('---------------------------------')
	print ('Accuracy Score :   ', score)
	print ('the prediction of y: ', clean_pred)
	print ('---------------------------------')

	total_asr = []
	
	for i in range(10):
		print ('The perturbation idx', i)
		perturb = np.array([float(line.rstrip('\n')) for line in open('../GUA/perturbation_results/{1}_xi4_epoch100/perturbation_{1}_{0}.txt'.format(i, args.dataset))])
		perturb = np.where(perturb>0.5, 1, 0)
		pt = np.where(perturb>0)[0].tolist()
		print ('The perturbations are', pt)
		asr_coll = []
		for j in range(len(idx_test)):
			neigh = list(graph.neighbors(idx_test[j]))
			print ('the neighrbors of node {} is'.format(j), neigh)
			tmp_G = graph.copy()
			# print ('the neighrbors of node {} is'.format(j), neigh)
			# print ('The edges in clean graph', tmp_G.number_of_edges())
			for k in pt:
				if k in neigh:
					print ('the node is', idx_test[j])
					# print ('the neighor is', k)
					# print ('the edges between', tmp_G.has_edge(idx_test[j],k))
					# print ('the edges between', tmp_G.has_edge(k, idx_test[j]))
					# print ('the edges of 12', tmp_G.has_edge(k,j))

					tmp_G.remove_edge(idx_test[j], k)
					# print ('the edges of 12', tmp_G.has_edge(k,j))
					# tmp_G.remove_edge(k, j)
				else:
					tmp_G.add_edge(idx_test[j], k)
					# tmp_G.add_edge(k, j)
			print ('The edges in attacked graph', tmp_G.number_of_edges())
			model = deepwalk(tmp_G, args) #the attacked graph embedding
			features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])
			X = features_matrix
			X_test = X[idx_test]
			# idx_pred = j - idx_test[0]
			att_pred = logisticRegr.predict(X_test)
			if clean_pred[j] == att_pred[j]:
				asr_coll.append(0)
				print ('attack fail', j)
			else:
				asr_coll.append(1)
				print ('attack success', j)
			print ('the success number', sum(asr_coll))
			print ('the total number', len(asr_coll))
		avg_asr = float(sum(asr_coll)/ len(asr_coll))
		print ('the asr is', avg_asr)
		total_asr.append(avg_asr)
	print ('the total asr over 10 experiments is:', total_asr)


	# for i in idx_test:
	#   folder_path = op.join("./" + args.dataset, "perturbation" + str(0))
	#   file_path = op.join(folder_path, "node" + str(i) + "_" + args.dataset + args.emb) 
	#   model = KeyedVectors.load_word2vec_format(file_path, binary=False)
	#   features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])
	#   X = features_matrix
	#   X_test = X[idx_test]
	#   idx_pred = i - idx_test[0]
		
	#   att_pred = logisticRegr.predict(X_test)
	#   if clean_pred[idx_pred] == att_pred[idx_pred]:
	#     asr_coll.append(0)
	#     print ('attack fail', idx_pred)
	#   else:
	#     asr_coll.append(1)
	#     print ('attack success', idx_pred)
	#   print ('the success number', sum(asr_coll))
	#   print ('the total number', len(asr_coll))
	# print ('the asr is', float(sum(asr_coll)/ len(asr_coll)))


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Start                                                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
if __name__ == "__main__":
	sys.exit(main())
