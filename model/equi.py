import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GraphNorm

import e3nn
from e3nn       import o3



class I2EConv(MessagePassing):
    """Equivariant message passing layer.

    Args:

        irreps_node (Irreps): Irreps of node attributes.
        irreps_edge (Irreps): Irreps of edge spherical harmonics.
        hidden_channels (int): Dim of invariant node features.
        out_channels (int): Dim of output node features.
    """
    def __init__(self,
        irreps_node,
        irreps_edge,
        hidden_channels,
        middle_channels,
        num_radial,
        aggr = 'mean',
    ):
        super().__init__(aggr = 'mean')

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge   = o3.Irreps(irreps_edge)

        self.middle_channels = middle_channels

        self.lin1 = Linear(2*hidden_channels+num_radial,middle_channels)

        edge_num_irreps = self.irreps_edge.num_irreps
        self.irreps_invariant     =o3.Irreps(f'{edge_num_irreps}x0e')
        self.lin2 = Linear(middle_channels,edge_num_irreps)

        self.conv = o3.ElementwiseTensorProduct(
                irreps_in1 = self.irreps_invariant,
                irreps_in2 = self.irreps_edge
              )

        self.irreps_out = self.conv.irreps_out
        print(self.conv)

        self.equi_sc = o3.FullTensorProduct(irreps_in1 =self.irreps_node,
               irreps_in2 = self.irreps_edge,
               filter_ir_out = ['0e','1e','2e'] )

        self.irreps_out = self.equi_sc.irreps_out
        print(self.irreps_out)
        self.dim = self.irreps_out.count('0e')
        print(self.dim)

        self.lin3 = Linear(hidden_channels+self.dim, hidden_channels)

        self.o3_lin = o3.Linear(self.irreps_out, self.irreps_edge)

    def forward(self, x, pos, edge_index, edge_attr, edge_pos):


        upd_pos = self.propagate(edge_index, x = x, edge_attr = edge_attr,edge_pos =edge_pos)

        pos = self.equi_sc(pos,upd_pos)

        x = torch.cat([x,pos[:,:self.dim]], dim =-1)
        x = self.lin3(x)

        pos = self.o3_lin(pos)
        # x, pos = torch.split(x, [self.middle_channels, self.irreps_out.dim], dim =-1)
        return x, pos, self.edge_pos

    def message(self, x_i, x_j,  edge_attr, edge_pos):

        edge_attr = self.lin1(torch.cat([x_i,x_j,edge_attr], dim =-1))

        edge_attr = F.silu(edge_attr)

        edge_weight = self.lin2(edge_attr)

        edge_pos = self.conv(F.silu(edge_weight), edge_pos)

        self.edge_pos = edge_pos

        # msg = torch.cat([edge_attr, edge_pos], dim =-1)

        return edge_pos