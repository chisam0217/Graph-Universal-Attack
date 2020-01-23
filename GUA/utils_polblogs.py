import numpy as np
from utils import load_data, load_polblogs_data
dataset = "polblogs"
if dataset == "polblogs":
    tmp_adj, features, labels, idx_train, idx_test = load_polblogs_data()
    print (sum(sum(tmp_adj)))
    print (tmp_adj.shape)
else:
    _, features, labels, idx_train, idx_val, idx_test, tmp_adj  = load_data(dataset)

for i in range(10):
    perturb = np.array([float(line.rstrip('\n')) for line in open("/home/xiao/Documents/pygcn/pygcn/vision4_result/{1}_xi4_epoch100/perturbation_{1}_{0}.txt".format(i, dataset))])
    idx = list(np.where(perturb>0.5))

    print (labels[idx])