import torch.nn as nn

from src.net.AttnMPNN import AttnMPNN
from src.net.MLP import MLP


class StackGNN(nn.Module):

    def __init__(self,
                 nf_dim: int,
                 ef_dim: int,
                 hnf_dim: int,
                 hef_dim: int,
                 nf_outdim: int,
                 ef_outdim: int,
                 n_layers: int):
        super(StackGNN, self).__init__()
        assert n_layers > 1
        self.gnn = AttnMPNN(node_in_dim=nf_dim,
                            edge_in_dim=ef_dim,
                            node_hidden_dim=hnf_dim,
                            edge_hidden_dim=hef_dim,
                            node_out_dim=hnf_dim,
                            edge_out_dim=hef_dim,
                            num_hidden_gn=n_layers - 1)  # it is the right one!
        self.node_nn = MLP(hnf_dim, nf_outdim,
                           num_neurons=[])

        self.edge_nn = MLP(hef_dim, ef_outdim,
                           num_neurons=[])

    def forward(self, g, nf, ef):
        """
        :param g: dgl.graph maybe batched
        :param nf: node feature; expected size [#. total nodes x 'raw' node feat dim]
        :param ef: edge feature; expected size [#. total edges x 'raw' edge feat dim]
        :return: unf, uef: updated node features, updated edge features
        """

        unf, uef = self.gnn(g, nf, ef)
        unf = self.node_nn(unf)
        uef = self.edge_nn(uef)
        return unf, uef
