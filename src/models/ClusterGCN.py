import torch
import torch.nn.functional as F
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv
from .gcnmlp import SAGEConv_MLP

# from ._GraphSampling import _GraphSampling
from .base import GraphSamplingBase
from utils import get_memory_usage, compute_tensor_bytes, MB, GB
import json
import time
from tqdm import tqdm


class ClusterGCN(GraphSamplingBase):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, data, train_idx, processed_dir):
        super(ClusterGCN, self).__init__(args, data, train_idx, processed_dir)

        if args.gnn_type == "gnn":
            base_gnnconv = SAGEConv
        elif args.gnn_type == "mlp":
            base_gnnconv = SAGEConv_MLP
        else:
            base_gnnconv = SAGEConv

        # build model
        self.convs = torch.nn.ModuleList()
        self.convs.append(base_gnnconv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(base_gnnconv(self.dim_hidden, self.dim_hidden))
        self.convs.append(base_gnnconv(self.dim_hidden, self.num_classes))

        # load data
        sample_size = max(
            1, int(args.batch_size / (data.num_nodes / args.num_parts)))
        cluster_data = ClusterData(
            data, num_parts=args.num_parts, recursive=False, save_dir=self.save_dir)
        self.train_loader = ClusterLoader(
            cluster_data, batch_size=sample_size, shuffle=True, num_workers=0)

        self.saved_args = vars(args)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
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
            out = self(batch.x, batch.edge_index)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                loss = loss_op(out[batch.train_mask],
                               batch.y[batch.train_mask])
            else:
                loss = loss_op(
                    out[batch.train_mask], batch.y[batch.train_mask].type_as(
                        out)
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(batch.y).sum())
            else:
                total_correct += int(out.eq(batch.y).sum())

        train_size = self.train_size if isinstance(
            loss_op, torch.nn.NLLLoss) else self.train_size * self.num_classes
        return total_loss / len(self.train_loader), total_correct / train_size
