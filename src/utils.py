# from texttable import Texttable
from torch_sparse import SparseTensor
import torch
import numpy as np
import scipy.sparse as sp

MB = 1024 ** 2
GB = 1024 ** 3


def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret




def random_sample_edges( adj, n, exclude):
    itr = sample_forever(adj, exclude=exclude)
    return [next(itr) for _ in range(n)]

def sample_forever( adj, exclude):
    """Randomly random sample edges from adjacency matrix, `exclude` is a set
    which contains the edges we do not want to sample and the ones already sampled
    """
    while True:
        # t = tuple(np.random.randint(0, adj.shape[0], 2))
        # t = tuple(random.sample(range(0, adj.shape[0]), 2))
        t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
        if t not in exclude:
            yield t
            exclude.add(t)
            exclude.add((t[1], t[0]))



