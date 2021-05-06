import torch
import torch.nn as nn

from src.net.AttnMPNNLayer import AttnMPNNLayer
from src.net.GRUMPNN import GRUMPNN
from src.net.MLP import MLP


class IJGNN(nn.Module):
    """
    Input injection GNN;
    """

    def __init__(self,
                 nf_dim: int,
                 ef_dim: int,
                 hnf_dim: int,
                 hef_dim: int,
                 nf_outdim: int,
                 ef_outdim: int):
        super(IJGNN, self).__init__()

        self.gnn = AttnMPNNLayer(node_in_dim=nf_dim + hnf_dim,
                                 edge_in_dim=ef_dim + hef_dim,
                                 node_out_dim=hnf_dim,
                                 edge_out_dim=hef_dim)

        self.node_nn = MLP(hnf_dim, nf_outdim,
                           num_neurons=[])

        self.edge_nn = MLP(hef_dim, ef_outdim,
                           num_neurons=[])

        self.hnf_dim = hnf_dim
        self.hef_dim = hef_dim

    def forward(self, g, nf, ef, n_iters: int):
        """
        :param g: dgl.graph maybe batched
        :param nf: node feature; expected size [#. total nodes x 'raw' node feat dim]
        :param ef: edge feature; expected size [#. total edges x 'raw' edge feat dim]
        :param n_iters: number of internal hops; related to the coverage of information propagation.
        :return: unf, uef: updated node features, updated edge features
        """

        assert n_iters >= 1
        nn, ne = nf.shape[0], ef.shape[0]

        hnf = torch.zeros(nn, self.hnf_dim, device=nf.device)
        hef = torch.zeros(ne, self.hef_dim, device=ef.device)

        for _ in range(n_iters):
            hnf = torch.cat([hnf, nf], dim=-1)
            hef = torch.cat([hef, ef], dim=-1)
            hnf, hef = self.gnn(g, hnf, hef)

        unf = self.node_nn(hnf)
        uef = self.edge_nn(hef)
        return unf, uef


class IJGNN2(nn.Module):
    """
    Maybe utilizing RNN like structure can be better?
    """

    def __init__(self,
                 nf_dim: int,
                 ef_dim: int,
                 hnf_dim: int,
                 hef_dim: int,
                 nf_outdim: int,
                 ef_outdim: int):
        super(IJGNN2, self).__init__()

        # gnn layer
        self.gnn = AttnMPNNLayer(node_in_dim=nf_dim,
                                 edge_in_dim=ef_dim,
                                 node_out_dim=hnf_dim,
                                 edge_out_dim=hef_dim)

        # RNNs - for smartly fusing the injected inputs and hidden embeddings
        self.node_rnn = nn.GRUCell(hnf_dim, hnf_dim)
        self.edge_rnn = nn.GRUCell(hef_dim, hef_dim)

        # decoders : transform hidden node/edge embeddings to the target
        self.node_nn = MLP(hnf_dim, nf_outdim,
                           num_neurons=[])

        self.edge_nn = MLP(hef_dim, ef_outdim,
                           num_neurons=[])

        self.hnf_dim = hnf_dim
        self.hef_dim = hef_dim

    def forward(self, g, nf, ef, n_iters: int):
        """
        :param g: dgl.graph maybe batched
        :param nf: node feature; expected size [#. total nodes x 'raw' node feat dim]
        :param ef: edge feature; expected size [#. total edges x 'raw' edge feat dim]
        :param n_iters: number of internal hops; related to the coverage of information propagation.
        :return: unf, uef: updated node features, updated edge features
        """

        assert n_iters >= 1
        nn, ne = nf.shape[0], ef.shape[0]

        hnf = torch.zeros(nn, self.hnf_dim, device=nf.device)
        hef = torch.zeros(ne, self.hef_dim, device=ef.device)

        for _ in range(n_iters):
            unf, uef = self.gnn(g, nf, ef)
            hnf = self.node_rnn(unf, hnf)
            hef = self.edge_rnn(uef, hef)

        unf = self.node_nn(hnf)
        uef = self.edge_nn(hef)
        return unf, uef


class IJGNN3(nn.Module):
    """
   Maybe utilizing RNN like structure can be better?
   """

    def __init__(self,
                 nf_dim: int,
                 ef_dim: int,
                 hnf_dim: int,
                 hef_dim: int,
                 nf_outdim: int,
                 ef_outdim: int,
                 n_iters: int):
        super(IJGNN3, self).__init__()

        # encoders
        self.node_encoder = nn.Linear(nf_dim, hnf_dim)
        self.edge_encoder = nn.Linear(ef_dim, hef_dim)

        # gnn layer
        self.gn = GRUMPNN(hnf_dim, hef_dim, n_iters)

        # decoders
        self.node_decoder = MLP(hnf_dim, nf_outdim, num_neurons=[])
        self.edge_decoder = MLP(hef_dim, ef_outdim, num_neurons=[])

    def forward(self, g, nf, ef):
        """
        :param g: dgl.graph maybe batched
        :param nf: node feature; expected size [#. total nodes x 'raw' node feat dim]
        :param ef: edge feature; expected size [#. total edges x 'raw' edge feat dim]
        :return: unf, uef: updated node features, updated edge features
        """

        unf, uef = self.node_encoder(nf), self.edge_encoder(ef)
        unf, uef = self.gn(g, unf, uef)
        unf, uef = self.node_nn(unf), self.edge_nn(uef)
        return unf, uef
