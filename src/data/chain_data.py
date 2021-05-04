import dgl
import numpy as np
import torch


def generate_graph(n_nodes: int, order: int, op='max'):
    src, dst = [], []
    for i in range(0, n_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    src = torch.tensor(src)
    dst = torch.tensor(dst)

    line_g = dgl.graph((src, dst), num_nodes=n_nodes)

    if op == 'max':
        op_ = max
    elif op == 'sum':
        op_ = sum
    elif op == 'average':
        op_ = np.average

    x = torch.rand(n_nodes, 1)
    y = []
    for i in range(n_nodes):
        start = max(0, i - order)
        end = min(n_nodes, i + order) + 1
        y.append(op_(x[start:end]))

    line_g.ndata['x'] = x
    line_g.ndata['y'] = torch.tensor(y).reshape(n_nodes, 1)
    line_g.edata['x'] = torch.ones(line_g.num_edges(), 1)
    return line_g


def generate_graphs_seq(n_graphs, nS_bd, order: int = 5, op='max'):
    nS = np.random.randint(nS_bd[0], nS_bd[1], size=n_graphs)
    gs = [generate_graph(ns, order, op) for ns in nS]
    return gs
