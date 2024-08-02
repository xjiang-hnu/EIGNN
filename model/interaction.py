import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GraphNorm

import e3nn
from e3nn       import o3

from equi import I2EConv
from invari import E2IConv_simple


class Interaction_e(torch.nn.Module):
    """
        irreps_node (Irreps or str): Irreps of node attributes.
        irreps_edge (Irreps or str): Irreps of edge spherical harmonics (constant througout model).
        irreps_out (Irreps or str): Irreps of output node features.
        num_radial (int): Dim of radial basis function.
    """
    def __init__(self,
        irreps_node,
        irreps_edge,
        hidden_channels,
        middle_channels,
        num_radial  = 8,
    ):
        super().__init__()


        self.I2EConv = I2EConv(irreps_node,
                    irreps_edge,
                    hidden_channels,
                    middle_channels,
                    num_radial)

        self.E2IConv = E2IConv_simple(irreps_node,
                    irreps_edge,
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    hidden_channels)


        self.norm = GraphNorm(hidden_channels)
        self.e_norm = e3nn.nn.BatchNorm(irreps_edge)

    def forward(self, x, pos, edge_index, rbf,edge_sh,batch):

        x, pos, edge_pos = self.I2EConv(x,pos, edge_index, rbf,edge_sh)

        m= self.E2IConv(x, pos, edge_index, rbf, edge_pos)
        x = x + m
        x = self.norm(x)
        pos = self.e_norm(pos)


        return x, pos

