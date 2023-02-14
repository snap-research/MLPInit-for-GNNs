import json
# from signal import raise_signal
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import NeighborSampler
# from torch_geometric.nn import SAGEConv
from .gcnmlp import SAGEConv_MLP, GATConv_MLP, GCNConv, GATConv, my_GCNConv, SAGEConv

from .gcnmlp import my_GCNConv as GCNConv
from .gcnmlp import my_GCNConv_MLP as GCNConv_MLP


# from torch_geometric.nn import GCNConv as SAGEConv
# from torch_geometric.nn import GraphConv as SAGEConv

from tqdm import tqdm

from utils import GB, MB, compute_tensor_bytes, get_memory_usage


# from ._GraphSampling import _GraphSampling
from .base import GraphSamplingBase


class GraphSAGE(GraphSamplingBase):
    def __init__(self, args, data, train_idx, processed_dir):
        super(GraphSAGE, self).__init__(args, data, train_idx, processed_dir)

        self.args = args

        if args.type_model == "GraphSAGE":
            if args.gnn_type == "gnn":
                base_gnnconv = SAGEConv
            elif args.gnn_type == "mlp":
                base_gnnconv = SAGEConv_MLP
            else:
                base_gnnconv = SAGEConv

        elif args.type_model == "GAT":
            if args.gnn_type == "gnn":
                base_gnnconv = GATConv
            elif args.gnn_type == "mlp":
                base_gnnconv = GATConv_MLP
            else:
                base_gnnconv = GATConv

        elif args.type_model == "GCN":
            if args.gnn_type == "gnn":
                base_gnnconv = GCNConv
            elif args.gnn_type == "mlp":
                base_gnnconv = GCNConv_MLP
            else:
                base_gnnconv = GCNConv

        else:
            raise NotImplementedError

        # build model
        self.convs = torch.nn.ModuleList()
        self.convs.append(base_gnnconv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(base_gnnconv(self.dim_hidden, self.dim_hidden))
        self.convs.append(base_gnnconv(self.dim_hidden, self.num_classes))

        # data load
        num_neighbors = [25, 10, 5, 5, 5, 5, 5, 5, 5]
        if self.args.type_model in ["GraphSAGE", "GAT", "GCN"]:
            self.train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=num_neighbors[: self.num_layers],
                                                batch_size=self.batch_size, shuffle=True, num_workers=12)
        else:
            raise NotImplementedError

        # reset_parameters
        self.saved_args = vars(args)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):

            x_target = x[: size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            # x = self.convs[i]( x_target,  edge_index )
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def train_net(self, input_dict):

        device = input_dict["device"]
        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        total_loss = total_correct = 0
        for batch_size, n_id, adjs in tqdm(self.train_loader):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.

            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = self(x[n_id], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            # out = out.type( torch.long )
            loss = loss_op(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-
                                     1).eq(y[n_id[:batch_size]]).sum())
            else:
                total_correct += int(out.eq(y[n_id[:batch_size]]).sum())

        train_size = (
            self.train_size
            if isinstance(loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )
        return total_loss / len(self.train_loader), total_correct / train_size
