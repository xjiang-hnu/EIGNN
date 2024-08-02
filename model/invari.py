import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GraphNorm

import e3nn
from e3nn       import o3



class E2IConv_simple(MessagePassing):
    """Equivariant feature scalarization layer and Invariantl meassage passing layer.
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
        out_channels,
    ):
        super().__init__(aggr = 'mean')

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge   = o3.Irreps(irreps_edge)

        ################

        self.conv = o3.ElementwiseTensorProduct(
                irreps_in1 = self.irreps_edge,
                irreps_in2 = self.irreps_edge,
                filter_ir_out=['0e']
              )
        
        self.norm = o3.Norm(irreps_in = self.irreps_node, squared = True)
        self.lin_cat = Linear(num_radial+self.irreps_node.num_irreps+self.conv.irreps_out.num_irreps, out_channels)


        self.lin_r = Linear(hidden_channels , out_channels)





    def forward(self, x, pos, edge_index, edge_attr,edge_pos):


        if isinstance(x, Tensor):
            x = (x, x)

        if isinstance(x, Tensor):
            pos = (pos, pos)


        x = self.propagate(edge_index, x = x, pos = pos,edge_attr = edge_attr,edge_pos = edge_pos)

        return x
    def message(self, x_j, pos_i, pos_j, edge_attr, edge_pos):

        # edge_left_2 = torch.cat([edge_pos, pos_i], dim =-1)
        # edge_right_2 = torch.cat([pos_i, pos_j], dim =-1)

        edge_pos1 = self.conv(edge_pos,pos_j)
        edge_pos2 = self.norm(pos_i - pos_j)

        edge_attr = self.lin_cat(torch.cat([edge_attr, edge_pos1, edge_pos2], dim =-1))


        return  edge_attr.sigmoid() * F.silu(self.lin_r(x_j))

