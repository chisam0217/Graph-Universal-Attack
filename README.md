# Graph Universal Attack

## Usage
* PyTorch 0.4 or 0.5 
* Python 2.7 or 3.6
* networkx, scipy, sklearn, numpy, pickle

## Train the attack model 

**Example:** python generate_perturbation.py --dataset cora --radius 4 \

*dataset: the network dataset you are going to attack* \
*radius: the radius of the l2 Norm Projection*

The verision of jupyter notebook is also supported as: universal_attack.ipynb

## Evaluate the test ASR
After finishing the training of the GUA, we then evaluate the test asr over the test nodes 

**Example:** python eval_baseline.py --dataset cora --radius 4 --evaluate_mode universal \

*dataset: the network dataset you are going to attack* \
*radius: the radius of the l2 Norm Projection*
*evaluate_mode* has five values: 
* "universal": graph universal attack
* "limitted_attack": limitted attack
* "global_random": global random attack
* "victim_attack": victim attack
* "universal_delete": randomly delete a part of nodes from the trained anchor nodes, to find the trade-off

The perturbation results trained by GUA when **radius = 4**, for each dataset: Cora, Citeseer and Pol.Blogs are also listed in "GUA/perturbation_results", which can be used directly for testing

The verision of jupyter notebook is also supported as: evaluate.ipynb

You can also validate the transferability on other embedding methods: \
**Node2vec**: python node2vec/evaluate_n2v --dataset cora
