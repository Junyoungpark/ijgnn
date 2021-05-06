import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from src.net.AttnMPNNLayer import get_aggregator
from src.net.MLP import MLP


class GRUMPNN(nn.Module):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 n_iters: int,
                 node_aggregator: str = 'sum',
                 mlp_params: dict = None):
        super(GRUMPNN, self).__init__()

        self.edge_model = nn.GRUCell(input_dim=2 * node_dim,
                                     hidden_size=edge_dim)
        self.node_model = nn.GRUCell(input_dim=edge_dim,
                                     hidden_size=node_dim)

        self.attn_model = MLP(edge_dim, 1, **mlp_params)
        self.node_aggr = get_aggregator(node_aggregator)
        self.n_iters = n_iters

    def forward(self, g, nf, ef):
        for _ in range(self.n_iters):
            nf, ef = self._forward(g, nf, ef)
        return nf, ef

    def _forward(self, g, nf, ef):
        # single step update

        with g.local_scope():
            g.ndata['h'] = nf
            g.edata['h'] = ef

            # perform edge update
            g.apply_edges(func=self.edge_update)

            # compute attention score
            g.edata['attn'] = dglnn.edge_softmax(g, self.attn_model(g.edata['uh']))

            # update nodes
            g.update_all(message_func=self.message_func,
                         reduce_func=self.node_aggr,
                         apply_node_func=self.node_update)

            unf, uef = g.ndata['uh'], g.edata['uh']
            return unf, uef

    def edge_update(self, edges):
        input = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        hidden = edges.data['h']
        updated_hidden = self.edge_model(input, hidden)
        return {'uh': updated_hidden}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['uh'] * edges.data['attn']}

    def node_update(self, nodes):
        input = nodes.data['agg_m']
        hidden = nodes.data['h']
        updated_hidden = self.node_model(input, hidden)
        return {'uh': updated_hidden}
