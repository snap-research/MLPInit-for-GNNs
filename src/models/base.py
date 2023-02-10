import os

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from tqdm import tqdm


class GraphSamplingBase(torch.nn.Module):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, data, train_idx, processed_dir):
        super(GraphSamplingBase, self).__init__()

        self.type_model = args.type_model
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
        self.test_loader = NeighborSampler(data.edge_index, sizes=[-1], node_idx=None,
                                        batch_size=1024, shuffle=False, num_workers = 12)
        # self.test_loader = NeighborSampler(data.edge_index, sizes=[-1], node_idx=None,
        #                                 batch_size=1024, shuffle=False)
        self.data = data
        self.args = args


    def inference(self, input_dict):

        if (self.args.dataset, self.args.type_model) in [
                                                        ("ogbn-papers100M", "GraphSAGE"),
                                                        ("ogbn-papers100M", "GraphSAINT"),
                                                        ("ogbn-papers100M", "ClusterGCN"),
                                                        ]:
            return self.inference_cpu( input_dict )
        else:
            return self.inference_gpu( input_dict )



    def inference_cpu(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in tqdm( self.test_loader ):
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        return x_all



    def inference_gpu(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        x_all = x_all.to(device)
        # print( "before x_all.shape:", x_all.shape ) 

        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in tqdm( self.test_loader ):
                edge_index, _, size = adj.to(device)
                x = x_all[n_id]
                # print( f"n_id.shape: {n_id.shape}" )
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

            # print( "x_all.shape:", x_all.shape ) 

        x_all = x_all.cpu() 
        return x_all





    @torch.no_grad()
    def inference_full_model(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        model = input_dict["model"]
        x_all = x_all.to(device)


        xs = []
        for batch_size, n_id, adjs in tqdm( self.test_loader ):
            adjs = adjs.to(device)
            # adjs = [adj.to(device) for adj in adjs]
            x = model(x_all[n_id], [adjs])
            xs.append(x.cpu())
        x_all = torch.cat(xs, dim=0)
            
        return x_all



    @torch.no_grad()
    def inference_full_batch(self, input_dict):

        model = input_dict["model"]
        device = input_dict["device"]
        # x_all = input_dict["x"]

        model.eval()
        model.set_aggr('mean')


        out = model(self.data.x.to(device), self.edge_index.to(device))

        return out

