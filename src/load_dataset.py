from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import os

import numpy as np
import torch
import torch_geometric.datasets
from torch_geometric.utils import subgraph, to_undirected




def load_data(dataset, dataset_dir='/home/xhan2/data/gnn_space'):

    if dataset == "ogbn-products":
        # root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")
        root = os.path.join(dataset_dir, dataset)
        dataset = PygNodePropPredDataset(name="ogbn-products", root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbn-products")
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset == "ogbn-arxiv":
        # root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")
        root = os.path.join(dataset_dir, dataset)
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbn-arxiv")
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)

        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()



    elif dataset in ["Reddit", "Flickr", "Yelp"]:
        path = os.path.join(dataset_dir, dataset)
        dataset_class = getattr(torch_geometric.datasets, dataset)
        dataset = dataset_class(path)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y


    elif dataset in ["AmazonProducts"]:
        path = os.path.join(dataset_dir, dataset)
        dataset_class = getattr(torch_geometric.datasets, dataset)
        dataset = dataset_class(path)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y.type(torch.float32)

    elif dataset in ["Reddit2"]:
        path = os.path.join(dataset_dir, dataset)
        dataset_class = getattr(torch_geometric.datasets, dataset)
        dataset = dataset_class(path)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y


    elif dataset in ["ogbn-proteins"]:
        root = os.path.join(dataset_dir, dataset)
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbn-proteins")
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()

    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")

    return data, x, y, split_masks, evaluator, processed_dir
