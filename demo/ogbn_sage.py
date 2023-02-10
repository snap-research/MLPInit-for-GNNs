# ten runs
# MLP init test accuracy: 0.8100 ± 0.0040
# Random init test accuracy: 0.7982 ± 0.0041

import os
import os.path as osp
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import torch
import torch.nn.functional as F

from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
import torch.utils.data as data_utils
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default='random', choices=['random', 'mlp'])
parser.add_argument('--dataset', type=str, default="ogbn-products", choices=["ogbn-products"])
parser.add_argument('--dataset_dir', type=str, default="")
args = parser.parse_args()
print( "args:", args )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = osp.join( args.dataset_dir, args.dataset)

dataset = PygNodePropPredDataset(args.dataset, root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=args.dataset)
data = dataset[0]
train_idx = split_idx['train']

X_train = data.x[ split_idx["train"] ]
y_train = data.y[ split_idx["train"] ].reshape(-1).type(torch.long)


x = data.x
y = data.y.squeeze()



print( "data.x.shape:", data.x.shape )
print( "data.y.shape:", data.y.shape )
print( "data.x.type:", x.dtype )
print( "data.y.type:", y.dtype )
print( "X_train.shape:", X_train.shape )
print( "y_train.shape:", y_train.shape )


y = data.y.squeeze().type(torch.long)
print( "data.y.type:", y.dtype )


X_y_train_mlpinit = data_utils.TensorDataset(X_train, y_train)
X_y_all_mlpinit = data_utils.TensorDataset(x, y)

train_mlpinit_loader = data_utils.DataLoader(X_y_train_mlpinit, batch_size=4096, shuffle=True, num_workers=12)
all_mlpinit_loader = data_utils.DataLoader(X_y_all_mlpinit, batch_size=4096, shuffle=False, num_workers=12)


class SAGEConv_like_MLP(torch.nn.Module):
   
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        # kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor]) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = x[1]
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out



class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv_like_MLP(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv_like_MLP(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv_like_MLP(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers):
            x_target = x
            x = self.convs[i]((x, x_target))
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x.log_softmax(dim=-1)




class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x.log_softmax(dim=-1)



    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



model_mlpinit = MLP(dataset.num_features, 512, dataset.num_classes, num_layers=4)
model_mlpinit = model_mlpinit.to(device)
optimizer_model_mlpinit = torch.optim.Adam(model_mlpinit.parameters(), lr=0.001, weight_decay = 0.0)



def train_mlpinit():
    model_mlpinit.train()
    total_loss = total_correct = 0

    for x, y in tqdm( train_mlpinit_loader ):

        x = x.to( device )
        y = y.to( device )

        optimizer_model_mlpinit.zero_grad()
        out = model_mlpinit(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer_model_mlpinit.step()

        total_loss += float(loss)

    loss = total_loss / len(train_mlpinit_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test_mlpinit():
    model_mlpinit.eval()

    out_list = []
    y_list = []

    for x, y in tqdm( all_mlpinit_loader ):
        x = x.to( device )
        y = y.to( device )
        out = model_mlpinit(x)
        out_list.append( out )
        y_list.append( y )


    out = torch.cat(out_list, dim=0)
    y_true = torch.cat(y_list, dim=0).cpu().unsqueeze(-1)


    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc



train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[25, 10, 5, 5], batch_size=1024,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)


model = SAGE(dataset.num_features, 512, dataset.num_classes, num_layers=4)
model = model.to(device)



def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        out = model(x[n_id].to(device), adjs)
        loss = F.nll_loss( out, y[n_id[:batch_size]].to(device) )
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq( y[n_id[:batch_size]].to(device) ).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc




@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1, 11):
    print('')
    print(f'Run {run:02d}:')
    print('')

    if args.init == "random":
        print( "using random init!" )
        model.reset_parameters()
    elif args.init == "mlp":
        print( "using MLP init!" )
        best_val_acc_init = 0
        model_mlpinit.reset_parameters()
        
        for epoch in range(1, 50):
            loss, acc = train_mlpinit()
            train_acc_init, val_acc_init, test_acc_init = test_mlpinit()
            print(  "train_acc_init, val_acc_init, test_acc_init:", train_acc_init, val_acc_init, test_acc_init )

            if val_acc_init > best_val_acc_init:
                best_val_acc_init = val_acc_init
                # print( train_acc_init, val_acc_init, test_acc_init )
                print( f"saving model_mlpinit at epcoh {epoch}" )
                torch.save(model_mlpinit.state_dict(), f'./model_mlpinit.pt' )

        model.load_state_dict(torch.load( f'./model_mlpinit.pt'  ))
    else:
        print( "input init method." )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 21):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 0:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
