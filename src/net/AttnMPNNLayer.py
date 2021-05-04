import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from src.net.MLP import MLP

AGGR_TYPES = ['sum', 'mean', 'max']


def get_aggregator(mode, from_field='m', to_field='agg_m'):
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = dgl.function.sum(from_field, to_field)
        if mode == 'mean':
            aggr = dgl.function.mean(from_field, to_field)
        if mode == 'max':
            aggr = dgl.function.max(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))
    return aggr


class AttnMPNNLayer(nn.Module):
    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 node_out_dim: int,
                 edge_out_dim: int,
                 node_aggregator: str = 'mean',
                 mlp_params: dict = {}):
        super(AttnMPNNLayer, self).__init__()
        self.edge_model = MLP(edge_in_dim + 2 * node_in_dim, edge_out_dim, **mlp_params)
        self.attn_model = MLP(edge_in_dim + 2 * node_in_dim, 1, **mlp_params)
        self.node_model = MLP(node_in_dim + edge_out_dim, node_out_dim, **mlp_params)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g, nf, ef):
        with g.local_scope():
            g.ndata['h'] = nf
            g.edata['h'] = ef

            # perform edge update
            g.apply_edges(func=self.edge_update)

            # compute attention score
            g.edata['attn'] = dglnn.edge_softmax(g, self.attn_model(g.edata['em_input']))

            # update nodes
            g.update_all(message_func=self.message_func,
                         reduce_func=self.node_aggr,
                         apply_node_func=self.node_update)

            unf, uef = g.ndata['uh'], g.edata['uh']
            return unf, uef

    def edge_update(self, edges):
        sender_nf = edges.src['h']
        receiver_nf = edges.dst['h']
        ef = edges.data['h']
        em_input = torch.cat([ef, sender_nf, receiver_nf], dim=-1)
        updated_ef = self.edge_model(em_input)
        return {'uh': updated_ef, 'em_input': em_input}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['uh'] * edges.data['attn']}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        nm_input = torch.cat([agg_m, nf], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}
