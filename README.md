
# MLPInit: Embarrassingly Simple GNN Training Acceleration with MLP Initialization

Implementation for ICLR2023 paper MLPInit: Embarrassingly Simple GNN Training Acceleration with MLP Initialization [[openreview]](https://openreview.net/forum?id=P8YIphWNEGO) [[arxiv]](https://arxiv.org/abs/2210.00102).

## 1. Introduction

Training graph neural networks (GNNs) on large graphs is complex and extremely time consuming. This is attributed to overheads caused by sparse matrix multiplication, which are sidestepped when training multi-layer perceptrons (MLPs) with only node features. MLPs, by ignoring graph context, are simple and faster for graph data, however they usually sacrifice prediction accuracy, limiting their applications for graph data. We observe that for most message passing-based GNNs, we can trivially derive an analog MLP (we call this a \peermlp) with an equivalent weight space, by setting the trainable parameters with the same shapes, making us curious about how do GNNs using weights from a fully trained \peermlp perform? Surprisingly, we find that GNNs initialized with such weights significantly outperform their PeerMLPs, motivating us to use PeerMLP training as a precursor, initialization step to GNN training. To this end, we propose an embarrassingly simple, yet hugely effective initialization method for GNN training acceleration, called MLPInit. Our extensive experiments on multiple large-scale graph datasets with diverse GNN architectures validate that MLPInit can accelerate the training of GNNs (up to 33× speedup on OGBN-Products) and often improve prediction performance (e.g., up to $7.97\%$ improvement for GraphSAGE across $7$ datasets for node classification, and up to $17.81\%$ improvement across $4$ datasets for link prediction on metric Hits@10).
### 1.1 The training speed comparison of the GNNs with Random initialization and MLPInit.
<img src="img/res.png" style="zoom:100%;" />

### 1.2 PyTorch-style Pseudocode of MLPInit
<img src="img/algo.png" style="zoom:40%;" />




## 2. Minimal Example

1. We provide the demo to try our proposed method, MLPInit.

```bash
## cmd for ogbn-products dataset
python demo/ogbn_sage.py --init random --dataset ogbn-products
python demo/ogbn_sage.py --init mlp    --dataset ogbn-products
```

2. We provide a [Jupyter Notebook](demo/demo.ipynb) to show the results on ogb-products dataset.


<img src="img/output.png" style="zoom:60%;" />




## 3. Environments

```
torch                   1.9.0
torch-geometric         2.0.4
ogb                     1.3.3
```

### 4. MLPInit

```bash
## cmd for ogbn-arxiv dataset on GraphSAGE
python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_gnn_type mlp --type_model GraphSAGE --weight_decay 0 --log_dir .
python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_gnn_type gnn --type_model GraphSAGE --weight_decay 0 --log_dir . --pretrained_checkpoint ./ogbn-arxiv_GraphSAGE_mlp_512_4_31415.pt 
```



## 5. Cite Our Paper

If you find our paper is useful for you research, please cite our paper.

```
@inproceedings{han2023mlpinit,
title={{MLPI}nit: Embarrassingly Simple {GNN} Training Acceleration with {MLP} Initialization},
author={Xiaotian Han and Tong Zhao and Yozen Liu and Xia Hu and Neil Shah},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=P8YIphWNEGO}
}
```


## 6. Others
Our code is based on https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking

