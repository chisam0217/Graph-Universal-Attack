from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, normalize, load_polblogs_data
from models import GAT, SpGAT
import math
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', type=str, default="citeseer", help='Dataset')
parser.add_argument('--evaluate_mode', type=str, default="universal", help='the attack method')
parser.add_argument('--radius', type=int, default=4, help='the radius of l2 norm')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == "polblogs":
    tmp_adj, features, labels, idx_train, idx_test = load_polblogs_data()
else:
    _, features, labels, idx_train, idx_val, idx_test, tmp_adj  = load_data(args.dataset)
num_classes = labels.max().item() + 1
# tmp_adj = tmp_adj.toarray()

adj = tmp_adj
adj = np.eye(tmp_adj.shape[0]) + adj
adj, _ = normalize(adj)
adj = torch.from_numpy(adj.astype(np.float32))

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    if args.dataset != "polblogs":
        idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
ori_output = compute_test()



def add_perturb(input_adj, idx, perturb):
    # (1-x)A + x(1-A)
#     input_adj = input_adj.toarray()

    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb
#     print ('x', x[idx])

    
#     x += np.transpose(x) #change the idx'th row and column
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
#     print ('x1', x1[idx])
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj
#     print ('adj2', adj2[idx])

    for i in range(input_adj.shape[0]):      
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj

def evaluate_attack(perturb):
    res = []
    # perturb = np.where(perturb>0.5, 1, 0)
    print ('perturb', perturb)
    new_pred = []
    for i in range(num_classes):
        new_pred.append(0)
    for k in idx_test:
#     for k in range(1):
        print ('test node', k)
        innormal_x_p = add_perturb(tmp_adj, k, perturb)
        print ('the perturbed conn', sum(innormal_x_p[k]))
#         innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)

#         diff = innormal_x_p[k] - tmp_adj[k]
#         diff_idx = np.where(diff != 0 )
        
#         print ('diff_idx', diff_idx)
    #     one_idx = np.where(innormal_x_p[k]==1)[0]
    #     zero_idx = np.where(innormal_x_p[k]!=1)[0]
    #     total_idx = one_idx.shape[0] + zero_idx.shape[0]
    #     print ('total_idx', total_idx)
    #     print ('one_idx', one_idx)
    #     print ('corresponding perturb', perturb[one_idx])
    #     print (innormal_x_p[k][one_idx])
        x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_adj.shape[0]))
        x_p = torch.from_numpy(x_p.astype(np.float32))
        x_p = x_p.cuda()
        output = model(features, x_p)
        new_pred[int(torch.argmax(output[k]))] += 1
        if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
            res.append(0)
        else:
            res.append(1)
    fooling_rate = float(sum(res)/len(res))
    print ('the current fooling rate is', fooling_rate)
    return fooling_rate, new_pred

def calculate_entropy(pred):
    h = 0
    all_pred = sum(pred)
    for i in range(num_classes):
        Pi = pred[i]/all_pred
        if Pi != 0:
            h -=  Pi* math.log(Pi)
    return h

new_pred = []
for i in range(num_classes):
    new_pred.append(0)
for k in idx_test:
    new_pred[int(torch.argmax(ori_output[k]))] += 1
entropy = calculate_entropy(new_pred)
print ('the entropy is', entropy)

#evaluate the universal attack
if args.evaluate_mode == "universal":
    fool_res = []
    p_times = []
    all_entropy = []
    for i in range(10):
        perturb = np.array([float(line.rstrip('\n')) for line in open('../GUA/perturbation_results/{1}_xi{2}_epoch100/perturbation_{1}_{0}.txt'.format(i, args.dataset, args.radius))])
        perturb = np.where(perturb>0.5, 1, 0)
        pt = np.where(perturb>0)[0]
        if len(list(pt)) == 0:
            fool_res.append(0)
            p_times.append(0)
            continue
        print ('the perturbation is', pt)
        res, new_pred = evaluate_attack(perturb)
        print ('the prediction result is', new_pred)
        entropy = calculate_entropy(new_pred)
        fool_res.append(res)      
        p_times.append(len(list(pt)))
        print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        print ('the entropy is', entropy)
        all_entropy.append(entropy)
    print ('all the entropy values are', all_entropy)
    print ('the average entropy is', sum(all_entropy)/float(len(all_entropy)))

