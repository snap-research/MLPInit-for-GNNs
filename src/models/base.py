import os

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from tqdm import tqdm


class GraphSamplingBase(torch.nn.Module):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, data, train_idx, processed_dir):
        super(GraphSamplingBase, self).__init__()

        self.gnn_model = args.gnn_model
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.train_size = train_idx.size(0)
        self.dropout = args.dropout
        self.train_idx = train_idx
        self.save_dir = processed_dir
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        #
        self.test_loader = NeighborSampler(
            data.edge_index,
            sizes=[-1],
            node_idx=None,
            batch_size=1024,
            shuffle=False,
            num_workers=12,
        )

        self.data = data
        self.args = args

    def inference(self, input_dict):
        return self.inference_gpu(input_dict)


    def inference_gpu(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        x_all = x_all.to(device)
        # print( "before x_all.shape:", x_all.shape )

        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in tqdm(self.test_loader):
                edge_index, _, size = adj.to(device)
                x = x_all[n_id]
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

        x_all = x_all.cpu()
        return x_all
