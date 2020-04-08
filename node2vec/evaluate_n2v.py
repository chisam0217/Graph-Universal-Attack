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
# import node2vec
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def node2vec(tmp_G, args):
	start_time = time.time()

	# Init graph
	G = graph.Graph(tmp_G, args.p, args.q)
	# G = graph.Graph(args.input, args.p, args.q)

	total_walks = G.G.number_of_nodes() * args.num_walks
	data_size = total_walks * args.walk_length

	print("\nNumber of nodes: {}".format(G.G.number_of_nodes()))
	print("Total number of walks: {}".format(total_walks)) # Number of walks starting from each node
	print("Data size (walks*length): {}\n".format(data_size))

	# Create the random walks and store them in walks list
	print("Generate walks ...")

	G.preprocess_transition_probs()
	walks = G.build_node2vec_walks(args.num_walks, args.walk_length)

	# Feed to walks to Word2Vec model
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimension, window=5, min_count=0, sg=1, workers=cpu_count(), iter=args.iter)

  	# Save to output file
	print("----- Total time {:.2f}s -----".format(time.time() - start_time))
	# model.wv.save_word2vec_format(args.output)
	return model.wv

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   EVALUATE                                                                                    #
#   Perform Logistic Regression of embedding                                                    #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
def main():
	parser = argparse.ArgumentParser()

	# parser.add_argument('--num-walks', default=10, type=int)
	# parser.add_argument('--walk-length', default=40, type=int)
	# parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
	# parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	# parser.add_argument('--model', default='word2vec', help='Type of model to apply on walks (word2vec/skipgram)')
	# # parser.add_argument("--emb", required=True)
	# parser.add_argument('--emb', default='.deepwalk.embeddings', help='the embedding file')
	# # parser.add_argument("--net", required=True)
	# # parser.add_argument("--labels", required=True)
	# parser.add_argument('--dic-network-name', default='network')
	# parser.add_argument('--dic-label-name', default='label')

	parser.add_argument('--walk-length', type=int, default=40)
	parser.add_argument('--num-walks', type=int, default=10)
	parser.add_argument('--dimension', type=int, default=128, help='Embeddings dimension')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter')
	parser.add_argument('--q', type=float, default=1, help='Input hyperparameter')
	parser.add_argument('--dataset', default='cora', help='dataset')

	args = parser.parse_args()

	#load dataset
	if args.dataset == "polblogs":
	    graph, _, labels, idx_train, idx_test = load_polblogs_data()
	else:

		graph, _, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

	## Load Embeddings
	# embeddings_file = op.join(args.dataset, args.dataset + args.emb)
	# model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	model = node2vec(graph, args)
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
	# for i in [0,7,8,9]:
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
					tmp_G.add_edge(idx_test[j], k, weight = 1)
					# tmp_G.add_edge(k, j)
			print ('The edges in attacked graph', tmp_G.number_of_edges())
			model = node2vec(tmp_G, args) #the attacked graph embedding
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


if __name__ == "__main__":
	sys.exit(main())
