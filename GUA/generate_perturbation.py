from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, normalize, load_polblogs_data
from models import GCN
from torch.autograd.gradcheck import zero_gradients
import os.path as op

# os.environ["CUDA_VISIBLE_DEVICES"]="1" 
# Training settings
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

args = parser.parse_args()




np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
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


# print (sum(features))
# print (labels.shape)
# print (idx_train.shape)
# print (idx_val.shape)
# print (idx_test)

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

def calculate_grad(pert_adj, idx, classes):
    x = Variable(pert_adj, requires_grad=True)
    output = model(features, x)
    grad = []

    for i in classes:
        cls = torch.LongTensor(np.array(i).reshape(1)).cuda()
        loss = F.nll_loss(output[idx:idx+1], cls) 
        loss.backward(retain_graph=True)
        grad.append(x.grad[idx].cpu().numpy())

    return np.array(grad)   


def add_perturb(input_adj, idx, perturb):
    # (1-x)A + x(1-A)
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb

    
 #change the idx'th row and column
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj

    for i in range(input_adj.shape[0]):   
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj

def proj_lp(v, xi=args.radius, p=2):
# def proj_lp(v, xi=8, p=2):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        v = v
    #################
    v = np.clip(v, 0, 1)
    ##################
    #to reduce the number of nonzero elements which means 
    #the times of perturbation, also prevents saddle point

    return v

def select_pert(pert_m):
    tmp_pert_m = np.absolute(pert_m)
    sort_idx = tmp_pert_m.argsort()[::-1]
    sel_idx = np.zeros(pert_m.shape[0])
    sel_idx[sort_idx[:args.pert_num]] = 1
    return sel_idx     

def convert_to_v(adj, pert_m, deg, idx):

    a = np.multiply(pert_m, deg)
    inv_m = np.ones(adj.shape[0]) - np.multiply(adj[idx], 2) 
    inv_m = np.power(inv_m, -1)
    res = np.multiply(a, inv_m)  
    return res

def normalize_add_perturb(ori_adj, idx, pert, d):
    a = ori_adj[idx] + pert
    inv_d = 1 + sum(pert)
    p_d = d * inv_d
    inv_d = 1.0/inv_d
    ## filter the perturbed matrix so that >= 0 
#     a = np.where(a<0, 0, a)
    ori_adj[idx] = np.multiply(a, inv_d)
    
    return ori_adj, p_d

def deepfool(innormal_adj, ori_adj, idx, num_classes, degree, overshoot=0.02, max_iter=30):
    #innormal_adj: the perturbed adjacency matrix not normalized
    #ori_adj: the normalized perturbed adjacency matrix 
    model.eval()
    pred = model(features, ori_adj)[idx]
    pred = pred.detach().cpu().numpy()
    
    I = pred.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]    
    f_i = np.array(pred).flatten()
    k_i = int(np.argmax(f_i))
    w = np.zeros(ori_adj.shape[0])
    r_tot = np.zeros(ori_adj.size(0))
    
    pert_adj = ori_adj.detach().cpu().numpy()
    pert_adj_tensor = ori_adj
    degree_idx = degree
    loop_i = 0
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        gradients = calculate_grad(pert_adj_tensor, idx, I)
        for i in range(1, num_classes):
            # set new w_k and new f_k
            w_k = gradients[i, :] - gradients[0, :]
            f_k = f_i[I[i]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i =  pert * w / np.linalg.norm(w)

        r_tot = r_tot + r_i
        

        pert_adj, _ = normalize_add_perturb(pert_adj, idx, (1+overshoot)*r_tot, degree_idx)
        #################
        pert_adj = np.clip(pert_adj, 0, 1)
        #################

        loop_i += 1
        
        # compute new label
        pert_adj_tensor = torch.from_numpy(pert_adj.astype(np.float32))
        pert_adj_tensor = pert_adj_tensor.cuda()
        f_i = np.array(model(features, pert_adj_tensor)[idx].detach().cpu().numpy()).flatten()
        k_i = int(np.argmax(f_i))
        if k_i != label:
            print ('attack succeed')
    
    r_tot = (1+overshoot)*r_tot
    print ('the r_tot', r_tot)
    r_tot = convert_to_v(innormal_adj, r_tot, degree_idx, idx)

    return r_tot, loop_i

def universal_attack(attack_epoch, max_epoch):
    model.eval()
    delta = 0.1
    fooling_rate = 0.0
    overshoot = 0.02
    # max_iter_df = 10
    max_iter_df = 30

    v = np.zeros(tmp_adj.shape[0]).astype(np.float32)

    cur_foolingrate = 0.0
    epoch = 0

    early_stop = 0
    results = []
    folder_path = op.join("./", "perturbation_results")
    if not op.exists(folder_path):
        os.mkdir(folder_path)

    while fooling_rate < 1 - delta and epoch < max_epoch:
        epoch += 1
        train_idx = idx_train.cpu().numpy()
        np.random.shuffle(train_idx)
        
        ###############################################
        print ('deepfooling...')
        attack_time = time.time()
        for k in train_idx:
            print ('deepfool node',k)
            #add v to see if the attack succeeds
            innormal_x_p = add_perturb(tmp_adj, k, v)
            ##################whether to use filtering
    #         innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)

            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_adj.shape[0])) #A' = A + I
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()

            output = model(features, x_p)
    #         print ('output', output[k])

            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                dr, iter = deepfool(innormal_x_p, x_p, k, num_classes, degree_p[k], overshoot, max_iter_df )
                if iter < max_iter_df-1:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v)

                else:
                    print ('cant attack this node')
            else:
                print ('attack succeed')
        print ('the deepfooling time cost is', time.time()-attack_time)

        print ('the perturbation matrix is', v)
        print ('testing the attack success rate')
        res = []
        #################
        v = np.where(v>0.5, 1, 0)
        ##################
        for k in train_idx:
            print ('test node', k)          
            innormal_x_p = add_perturb(tmp_adj, k, v)        
            ############    
            # innormal_x_p = np.where(innormal_x_p<0.5, 0, 1)
            ############
            
            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_adj.shape[0]))
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()
            output = model(features, x_p)
            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                res.append(0)
            else:
                res.append(1)
        fooling_rate = float(sum(res)/len(res))
        print ('the current fooling rate is', fooling_rate)

        ####################
        # if fooling_rate > cur_foolingrate:
        #####################
        if fooling_rate >= cur_foolingrate:
            cur_foolingrate = fooling_rate
            file_path = op.join(folder_path, '{1}_xi{2}_epoch100/perturbation_{1}_{0}.txt'.format(attack_epoch, args.dataset, args.radius))
            with open(file_path, "w") as f:
                for i in v:
                    f.write(str(i) + '\n')
#################
        results.append(fooling_rate)
        if epoch > 3:
            if fooling_rate == results[-2]:
                early_stop += 1
            else:
                early_stop = 0
        if early_stop == 15:
            break
#####################
    return cur_foolingrate
        

train_foolrate = []
for i in range(0,10):
    fool_rate = universal_attack(i, 100)
    train_foolrate.append(fool_rate)
print ('the final train fool rate', train_foolrate)                                                       
