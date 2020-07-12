from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, normalize, load_polblogs_data
from models import GCN

parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='The name of the network dataset.')
parser.add_argument('--radius', type=int, default=4,
                    help='The radius of l2 norm projection')
parser.add_argument('--evaluate_mode', type=str, default="universal",
                    help='The universal attack method.')
args = parser.parse_args()




np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == "polblogs":
    tmp_adj, features, labels, idx_train, idx_test = load_polblogs_data()
    print (sum(sum(tmp_adj)))
    print (tmp_adj.shape)
else:
    _, features, labels, idx_train, idx_val, idx_test, tmp_adj  = load_data(args.dataset)

num_classes = labels.max().item() + 1
# tmp_adj = tmp_adj.toarray()

adj = tmp_adj
adj = np.eye(tmp_adj.shape[0]) + adj
adj, _ = normalize(adj)
adj = torch.from_numpy(adj.astype(np.float32))

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    if args.dataset != "polblogs":
        idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    x = Variable(adj, requires_grad=True)
    output = model(features, x)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    optimizer.step()


    if args.dataset != "polblogs": 
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - t))


def test(adj_m):
    model.eval()
    output = model(features, adj_m)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
ori_output = test(adj)

def add_perturb(input_adj, idx, perturb):
    # (1-x)A + x(1-A)
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj

    for i in range(input_adj.shape[0]):      
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj

def evaluate_attack(perturb):
    res = []
    print ('perturb', perturb)
    new_pred = []
    for i in range(num_classes):
        new_pred.append(0)
    for k in idx_test:
        innormal_x_p = add_perturb(tmp_adj, k, perturb)
        x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_adj.shape[0]))
        x_p = torch.from_numpy(x_p.astype(np.float32))
        x_p = x_p.cuda()
        output = model(features, x_p)
        new_pred[int(torch.argmax(output[k]))] += 1
        if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
            res.append(0)
            print ('node {} attack failed'.format(k))
        else:
            res.append(1)
            print ('node {} attack succeed'.format(k))
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
        perturb = np.array([float(line.rstrip('\n')) for line in open('./perturbation_results/{1}_xi{2}_epoch100/perturbation_{1}_{0}.txt'.format(i, args.dataset, args.radius))])
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
        
elif args.evaluate_mode == "global_random":
    #set this equal to the ceil of the number of anchor nodes computed by universal attack
    perturb_times = 8
    fool_res = []
    p_times = []
    for i in range(10):
        prob = float(perturb_times / tmp_adj.shape[0])
        perturb = np.random.choice(2, tmp_adj.shape[0], p = [1-prob, prob])
        print ('the prob is', prob)
        pt = np.where(perturb>0)[0]
        print ('the perturbation is', pt)
        res, new_pred = evaluate_attack(perturb)
        fool_res.append(res)

        p_times.append(len(list(pt)))
        print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))

    print ('the average fooling rate with {} perturbation times is'.format(perturb_times), sum(fool_res)/float(len(fool_res)))
    
elif args.evaluate_mode == "limitted_random":
    
    perturb_times = 8
    fool_res = []
    p_times = []
    for i in range(10):
        perturb = np.zeros(adj.shape[1])
        attack_index = list(np.random.choice(range(adj.shape[1]), perturb_times, replace = False))
        perturb[attack_index] = 1
        pt = np.where(perturb>0)[0]
        res, new_pred = evaluate_attack(perturb)
        fool_res.append(res)

        p_times.append(len(list(pt)))
        print ('the perturbation times is', p_times)
        print ('the fooling rates are', fool_res)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))

    print ('the average fooling rate with {} perturbation times is'.format(perturb_times), sum(fool_res)/float(len(fool_res)))
    
elif args.evaluate_mode == "victim_attak":
    #the perturbation times of our universal perturbation
    perturb_time = 8 #set this equal to the ceil of the number of anchor nodes computed by universal attack
    fool_res = []
    for k in range(num_classes):
        each_fool_res = []
        idx = np.where(labels.cpu().numpy()==k)[0]
        for i in range(10):
            attack_index = list(np.random.choice(idx, perturb_time, replace = False))
            perturb = np.zeros(adj.shape[1])
            perturb[attack_index] = 1
            print ('perturbating by connecting to nodes of class', k)
            res, new_pred = evaluate_attack(perturb)
            each_fool_res.append(res)
            print ('the fooling rates of current class are', each_fool_res)
            avg_asr = sum(each_fool_res)/float(len(each_fool_res))
            print ('the average fooling rates over 10 times of test is', avg_asr)
        fool_res.append(avg_asr)
        print ('fool_res', fool_res)
    print ('the avg asr by connecting to each class of nodes is', fool_res)
        
elif args.evaluate_mode == "universal_delete":   
    
    all_fool = []
    for i in range(8, 9):
        fool_res = []
        for j in range(8):
            perturb = np.array([float(line.rstrip('\n')) for line in open('./perturbation_results/{1}_xi{2}_epoch100/perturbation_{1}_4.txt'.format(i, args.dataset, args.radius))])
            perturb = np.where(perturb>0.5, 1, 0)
            pt = np.where(perturb>0)[0]
            a = list(np.random.choice(range(0, pt.shape[0]), i, replace = False))
            perturb[pt[a]] = 0
            pt = np.where(perturb>0)[0]
            print ('the perturbation is', pt)
            res, new_pred = evaluate_attack(perturb)
            fool_res.append(res)      
            print ('the fooling rates are', fool_res)
            print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        all_fool.append(sum(fool_res)/float(len(fool_res)))
          
    print ('all the fooling rate is', all_fool)


