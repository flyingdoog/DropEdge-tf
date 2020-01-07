Unofficial TF version: DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
====
This is an unofficial Tensorflow implementation of the paper: DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. I simply add the random sampling part to the orignal GCN.


## Requirements

  * Python 3.6.2
  * tensorflow (>0.12)
  * networkx

## Usage
	*python train.py --dataset cora


## New Parameters
	* percent: sampling percent
	* normalization: normalization of adjacency matrix.
	* task_type: semi or full.   semi uses 120 nodes for trainning in citeseer and 140 nodes in cora.


## References
```
@inproceedings{
rong2020dropedge,
title={DropEdge: Towards Deep Graph Convolutional Networks on Node Classification},
author={Yu Rong and Wenbing Huang and Tingyang Xu and Junzhou Huang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Hkx1qkrKPr}

}

@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```




