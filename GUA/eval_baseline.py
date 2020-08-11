#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


class args:
    cuda = True
    fastmode = False
    seed = 20
#     seed = 123
    epochs = 200
    lr = 0.01
    weight_decay = 5e-4
    hidden = 16
    dropout = 0.5
    pert_num = 20
    L1 = 0.01
    evaluate_mode = "universal"
    dataset = "cora"
    radius = 4


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# In[3]:


if args.dataset == "polblogs":
    tmp_adj, features, labels, idx_train, idx_val, idx_test = load_polblogs_data()
    print (sum(sum(tmp_adj)))
    print (tmp_adj.shape)
else:
    _, features, labels, idx_train, idx_val, idx_test, tmp_adj  = load_data(args.dataset)

num_classes = labels.max().item() + 1
# tmp_adj = tmp_adj.toarray()

adj = tmp_adj
adj = np.eye(tmp_adj.shape[0]) + adj
adj, deg = normalize(adj)
adj = torch.from_numpy(adj.astype(np.float32))


# In[4]:


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# In[5]:


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    if args.dataset != "polblogs":
        idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# In[6]:


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    x = Variable(adj, requires_grad=True)
    output = model(features, x)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
#     print ('output', output.size())
#     print ('labels', labels.size())
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


# In[7]:


def test(adj_m):
    model.eval()
    output = model(features, adj_m)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output


# In[8]:


t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# torch.save(model, './cora_gcn.pth')
# torch.save(model.state_dict(), 'cora_gcn.pkl')

# Testing
ori_output = test(adj)


# In[9]:


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


# In[10]:


def evaluate_attack(perturb):
    res = []
    # perturb = np.where(perturb>0.5, 1, 0)
    print ('perturb', perturb)
    new_pred = []
    idx = np.where(perturb>0.5)[0].tolist()
    perturb_idx = labels.cpu().numpy()[idx]
    most_freq_idx = np.bincount(perturb_idx).argmax()
    
    succ_deg = []
    fail_deg = []
    node_belong_vc = []
    pred_belong_vc = []
    
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
            fail_deg.append(deg[k])
            print ('node {} attack failed'.format(k))
            if int(torch.argmax(ori_output[k])) == most_freq_idx:
                node_belong_vc.append(0)
            
        else:
            res.append(1)
            succ_deg.append(deg[k])
            print ('node {} attack succeed'.format(k))
            if int(torch.argmax(ori_output[k])) == most_freq_idx:
                node_belong_vc.append(1)
            if int(torch.argmax(output[k])) == most_freq_idx:
                pred_belong_vc.append(1)
        
            
    fooling_rate = float(sum(res)/len(res))
    print ('the current fooling rate is', fooling_rate)
    succ_deg = sum(succ_deg)/len(succ_deg)
    fail_deg = sum(fail_deg)/len(fail_deg)
    asr_vc = sum(node_belong_vc)/len(node_belong_vc)
    asr_nonvc = (sum(res) - sum(node_belong_vc))/(len(res) - len(node_belong_vc))
    pred_belong_vc = sum(pred_belong_vc) / sum(res)
    return fooling_rate, new_pred, succ_deg, fail_deg, asr_vc, asr_nonvc, pred_belong_vc


# In[11]:


def calculate_entropy(pred):
    h = 0
    all_pred = sum(pred)
    for i in range(num_classes):
        Pi = pred[i]/all_pred
        if Pi != 0:
            h -=  Pi* math.log(Pi)
    return h


# In[12]:


new_pred = []
for i in range(num_classes):
    new_pred.append(0)
for k in idx_test:
    new_pred[int(torch.argmax(ori_output[k]))] += 1
entropy = calculate_entropy(new_pred)
print ('the entropy is', entropy)


# In[13]:


#evaluate the universal attack
if args.evaluate_mode == "universal":
    
    for j in range(6, 13):
        fool_res = []
        p_times = []
        all_entropy = []

        _succ_deg = []
        _fail_deg = []
        _asr_vc = []
        _asr_nonvc = []
        _pred_belong_vc = []

        for i in range(10):
#             perturb = np.array([float(line.rstrip('\n')) for line in open('./perturbation_results/{1}_xi{2}_epoch100/perturbation_{1}_{0}.txt'.format(i, args.dataset, args.radius))])
            perturb = np.array([float(line.rstrip('\n')) for line in open('./perturbation_results/{1}_xi{2}_epoch100/perturbation_{1}_{0}.txt'.format(i, args.dataset, j))])
            perturb = np.where(perturb>0.5, 1, 0)
            pt = np.where(perturb>0)[0]
            if len(list(pt)) == 0:
                fool_res.append(0)
                p_times.append(0)
                continue

            res, new_pred, succ_deg, fail_deg, asr_vc, asr_nonvc, pred_belong_vc = evaluate_attack(perturb)
            fool_res.append(res)      
            p_times.append(len(list(pt)))

            _succ_deg.append(succ_deg)
            _fail_deg.append(fail_deg)
            _asr_vc.append(asr_vc)
            _asr_nonvc.append(asr_nonvc)
            _pred_belong_vc.append(pred_belong_vc)
            print (_succ_deg, _fail_deg, _asr_vc, _asr_nonvc, _pred_belong_vc)

        print ('the fooling rates are', fool_res)
        print ('the average fooling rates over 10 times of test is', sum(fool_res)/float(len(fool_res)))
        print ('the perturbation times is', p_times)

        print ('the average degree of successfully attacked nodes is', sum(_succ_deg)/len(_succ_deg))
        print ('the average degree of failed attacked nodes is', sum(_fail_deg)/len(_fail_deg))
        print ('the average ASR of nodes belong to victim class is', sum(_asr_vc)/len(_asr_vc))
        print ('the average ASR of nodes not belong to victim class is', sum(_asr_nonvc)/len(_asr_nonvc))
        print ('the rate of successfully attacked nodes that are misclassified into the victim class is', sum(_pred_belong_vc)/len(_pred_belong_vc))
    #         print ('all the entropy values are', all_entropy)
    #         print ('the average entropy is', sum(all_entropy)/float(len(all_entropy)))
        
elif args.evaluate_mode == "global_random":
    #set this equal to the ceil of the number of anchor nodes computed by universal attack
    perturb_times = 6
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

elif args.evaluate_mode == "advanced_limitted_random":
    perturb_times = 26
    fool_res = []
    p_times = []
    node_pool = np.argsort(deg)[-120:]
    for i in range(10):
        perturb = np.zeros(adj.shape[1])
        
        attack_index = list(np.random.choice(node_pool, perturb_times, replace = False))
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
#     p_times = []
    for k in range(num_classes):
#     for k in range(4,6):
        each_fool_res = []
        idx = np.where(labels.cpu().numpy()==k)[0]
#         for i in range(1):
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
   
elif args.evaluate_mode == "advanced_victim_attak":
    perturb_time = 8 #set this equal to the ceil of the number of anchor nodes computed by universal attack
    fool_res = []
#     p_times = []
    for k in range(num_classes):
        #select the nodes with the highest probabilities from the victim class k
        output_class = torch.argmax(ori_output, dim=1).detach().cpu().numpy()
        vc_nodes = np.where(output_class==k, 1, 0)
        vc_nodes = np.nonzero(vc_nodes)[0]
#         vc_nodes = np.where(labels.cpu().numpy()==k)[0]
        vc_probs = ori_output.detach().cpu().numpy()[vc_nodes, k]
        vc_indices = vc_probs.argsort()[-perturb_time:]
        vc_indices = vc_nodes[vc_indices]
        print ('Victim Nodes', vc_indices)
        perturb = np.zeros(adj.shape[1])
        perturb[vc_indices] = 1
        res, new_pred = evaluate_attack(perturb)
        fool_res.append(res)
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


# 
