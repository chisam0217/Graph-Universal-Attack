# Graph Universal Attack

## Usage
* PyTorch 0.4 or 0.5 
* Python 2.7 or 3.6

## Train the attack model

**Example:** python generate_perturbation --dataset cora --radius 4 \
*dataset: the network dataset you are going to attack* \
*radius: the radius of the l2 Norm Projection*

The verision of jupyter notebook is also supported as: universal_attack.ipynb

## Evaluate the test ASR
After finishing the training of the GUA, we then evaluate the test asr over the test nodes \

**Example:** python eval_baseline --dataset cora --radius 4 --evaluate_mode universal \
*evaluate_mode* has four values: 
* "universal": graph universal attack
* "limitted_attack": limitted attack
* "victim_attack": victim attack
* "universal_delete": randomly delete a part of nodes from the trained anchor nodes, to find the trade-off

The verision of jupyter notebook is also supported as: evaluate.ipynb
