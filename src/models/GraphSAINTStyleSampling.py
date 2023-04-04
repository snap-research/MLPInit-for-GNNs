import json
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler

from torch_geometric.nn import SAGEConv
from .PeerMLP import SAGEConvMLP

from torch_geometric.utils import degree

# from ._GraphSampling import _GraphSampling
from tqdm import tqdm

from .base import GraphSamplingBase


class GraphSAINT(GraphSamplingBase):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, data, train_idx, processed_dir):
        super(GraphSAINT, self).__init__(args, data, train_idx, processed_dir)
        self.use_norm = args.use_norm
        self.dropout = args.dropout
        self.args = args

        if args.gnn_type == "gnn":
            base_gnnconv = SAGEConv
        elif args.gnn_type == "mlp":
            base_gnnconv = SAGEConvMLP
        else:
            base_gnnconv = SAGEConv

        # build model
        self.convs = torch.nn.ModuleList()
        self.convs.append(base_gnnconv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(base_gnnconv(self.dim_hidden, self.dim_hidden))
        self.convs.append(base_gnnconv(self.dim_hidden, self.num_classes))
        # self.lin = Linear(self.num_layers * self.dim_hidden, self.num_classes)

        # data load
        row, col = self.edge_index = data.edge_index
        data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]

        self.train_loader = RWSampler(
            data,
            self.batch_size,
            args.walk_length,
            num_steps=args.num_steps,
            save_dir=self.save_dir,
            sample_coverage=args.sample_coverage,
            num_workers=0,
        )

        # reset_parameters
        self.reset_parameters()
        self.saved_args = vars(args)

        aggr = "add" if self.use_norm else "mean"
        self.set_aggr(aggr)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def train_net(self, input_dict):

        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        total_loss = total_correct = 0
        for batch in tqdm(self.train_loader):
            batch = batch.to(device)
            if batch.train_mask.sum() == 0:
                continue
            optimizer.zero_grad()
            if self.use_norm:
                edge_weight = batch.edge_norm * batch.edge_weight
                out = self(batch.x, batch.edge_index, edge_weight)
            else:
                out = self(
                    batch.x,
                    batch.edge_index,
                )

            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                if self.use_norm:
                    loss = F.nll_loss(out, batch.y, reduction="none")
                    loss = (loss * batch.node_norm)[batch.train_mask].sum()
                else:
                    loss = loss_op(out[batch.train_mask], batch.y[batch.train_mask])
            else:
                loss = loss_op(
                    out[batch.train_mask],
                    batch.y[batch.train_mask].type_as(out),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            optimizer.step()

            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(batch.y).sum())
            else:
                total_correct += int(out.eq(batch.y).sum())

        train_size = self.train_size if isinstance(loss_op, torch.nn.NLLLoss) else self.train_size * self.num_classes

        return total_loss / len(self.train_loader), total_correct / train_size
