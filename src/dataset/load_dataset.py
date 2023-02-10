from cmath import log
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import os

import numpy as np
import torch
import torch_geometric.datasets
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data, InMemoryDataset, download_url

from torch_geometric.utils import subgraph, to_undirected

import logging
logger = logging.getLogger()


from .non_homo_dataset import load_nc_dataset



def sample_subgraph( data, x, y, split_masks, evaluator, processed_dir, ratio = 0.1 ):

    subgrpah_train_mask_idx = split_masks["train"].nonzero().numpy().ravel()
    subgrpah_train_mask_idx = np.random.choice(subgrpah_train_mask_idx, size=int( ratio * len( subgrpah_train_mask_idx ) ), replace=False)
    # print( len(subgrpah_train_mask_idx) )
    subgrpah_train_mask = np.zeros(data.x.shape[0], dtype=int)
    subgrpah_train_mask[subgrpah_train_mask_idx] = 1
    subgrpah_train_mask = torch.tensor( subgrpah_train_mask.astype( np.bool8 ) )

    subgraph_mask = split_masks["valid"] + split_masks["test"] + subgrpah_train_mask

    edge_index, edge_attr = subgraph( subgraph_mask, data.edge_index, relabel_nodes=True  )

    data.x = data.x[ subgraph_mask ]
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.train_mask = split_masks["train"][subgraph_mask]
    data.valid_mask = split_masks["valid"][subgraph_mask]
    data.test_mask = split_masks["test"][subgraph_mask]

    data.y = data.y[subgraph_mask]
    data.num_nodes = data.x.shape[0]

    x = data.x
    y = data.y

    split_masks = {}
    split_masks["train"] = data.train_mask
    split_masks["valid"] = data.valid_mask
    split_masks["test"] = data.test_mask

    return data, x, y, split_masks, evaluator, processed_dir 
    


def load_data(dataset, dataset_dir = '/home/xhan2/data/gnn_space', ratio = None):

    if dataset == "Products":
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

        if ratio:
            data, x, y, split_masks, evaluator, processed_dir = sample_subgraph( data, x, y, split_masks, evaluator, processed_dir, ratio = ratio )

        logging.info( f"data: {data}" )

    elif dataset == "ogbn-papers100M":
        # root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")
        root = os.path.join(dataset_dir, dataset)
        dataset = PygNodePropPredDataset(name="ogbn-papers100M", root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbn-papers100M")
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()

        if ratio:
            if ratio < 1:
                data, x, y, split_masks, evaluator, processed_dir = sample_subgraph( data, x, y, split_masks, evaluator, processed_dir, ratio = ratio )
        logging.info( f"data: {data}" )



    elif dataset == "ogbn-arxiv":
        # root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")
        root = os.path.join(dataset_dir, dataset)
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbn-arxiv")
        data = dataset[0]
        data.edge_index = to_undirected( data.edge_index  )

        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()

        if ratio:
            if ratio < 1:
                data, x, y, split_masks, evaluator, processed_dir = sample_subgraph( data, x, y, split_masks, evaluator, processed_dir, ratio = ratio )
        logging.info( f"data: {data}" )




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
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)
        logging.info( f"data: {data}" )



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
        y = data.y.type( torch.float32 )
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)

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
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)

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

        

    elif dataset in ["arxiv-year", "fb100", "wiki", "pokec", "snap-patents", "twitch-gamer", "genius"]:
        # root = os.path.join(dataset_dir, dataset)
        dataset = load_nc_dataset( dataname= dataset )
        processed_dir = None
        evaluator = None
        split_idx = dataset.get_idx_split()

        # evaluator = Evaluator(name="ogbn-proteins")
        evaluator = None
        data = Data()
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(dataset.graph["num_nodes"], dtype=torch.bool)
            mask[split_idx[split]] = True
            # print( mask )
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]

        data.edge_index = dataset.graph[ "edge_index" ]
        data.edge_index = to_undirected( data.edge_index  )
        data.num_nodes = dataset.graph[ "num_nodes" ]
        data.x = dataset.graph[ "node_feat" ] 
        data.y = dataset.label.reshape(-1)

        x = data.x
        y = data.y = data.y.squeeze()

    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")

    logger.info( f"data.x.shape: {x.shape}" )
    logger.info( f"data.y.shape: {y.shape}"  )
    logger.info( f"data.edge_index.shape: {data.edge_index.shape}" )

    return data, x, y, split_masks, evaluator, processed_dir

