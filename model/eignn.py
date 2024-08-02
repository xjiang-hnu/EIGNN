import torch
from torch.nn   import ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP
import e3nn
from e3nn       import o3
from e3nn.math  import soft_one_hot_linspace
from interaction import Interaction_e
from decoder import none_to_node_layer

class EIGNN(torch.nn.Module):
    """
    Args:
        
        irreps_node (Irreps or str): Irreps of node attributes.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        hidden_channels (int): Dimension of hidden layer.
        middle_channels (int): Dimension of middle layer.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        num_radial (int): Number of the radial basis functions.
        max_radius (float): Cutoff radius used during graph construction.
    """
    def __init__(self,
        irreps_node ='1x0e + 1x1e + 1x2e' ,
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        hidden_channels = 64,
        middle_channels =64,
        num_convs      = 4,
        num_radial = 8,
        out_channels = 20,
        max_radius     = 2.5,
    ):
        super().__init__()

        self.irreps_edge    = o3.Irreps(irreps_edge)
        # self.irreps_index = irreps_index
        self.num_convs      = num_convs
        self.max_radius     = max_radius
        self.num_edge_basis = num_radial


        self.interactions = ModuleList()
        self.lin_embedding = Linear(4, hidden_channels)
        self.lin_cat = Linear(2*hidden_channels, hidden_channels)
        for j in range(num_convs):

            conv = Interaction_e(
                irreps_node,
                  irreps_edge,
                  hidden_channels,
                  middle_channels,
                  num_radial  = 2*num_radial + 1,
            )

            self.interactions.append(conv)


        self.out = MLP([hidden_channels, hidden_channels,out_channels],bias = False)
        self.agg_3 = none_to_node_layer(MLP([2* hidden_channels, hidden_channels],dropout=0.3), 'mean')



    def forward(self, data):
        x = data.x.to(torch.long)
        batch = data.batch
        # edge_vec =  data.edge_attr.to(torch.float32)
        vec = data.pos.to(torch.float32)
        batch = data.batch
        edge_index = data.edge_index.to(torch.long)

        x = F.one_hot(x-1,  2).to(torch.float32)

        cross_pos_in = data.edge_attr_in
        cross_pos_inh = data.edge_attr_inh

        pe = data.pe.to(torch.float32)
        norm_in = cross_pos_in.norm(dim=1)
        norm_inh = cross_pos_inh.norm(dim=1)


        rbf2 = soft_one_hot_linspace(
            norm_in,
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'bessel',
            cutoff = True,
        )

        rbf = soft_one_hot_linspace(
            norm_inh,
            start  = 0.0,
            end    = self.max_radius,
            number = self.num_edge_basis,
            basis  = 'bessel',
            cutoff = True,
        )

        edge_sh = o3.spherical_harmonics(self.irreps_edge, cross_pos_in, normalize=True, normalization='component')

        pos = o3.spherical_harmonics(self.irreps_edge, vec, normalize=True, normalization='component')

        rbf = torch.cat([(norm_inh-norm_in).unsqueeze(-1), rbf, rbf2], dim =-1)
        x = torch.cat([x, pe.unsqueeze(-1),vec.norm(dim =-1, keepdim = True)], dim= -1)
        x = self.lin_embedding(x)
        m=x
        
        ### Interaction layer
        for i, layer in enumerate(self.interactions):

          m, pos = layer(m,pos, edge_index, rbf,edge_sh,batch)

        ### Decode layer
        x = self.lin_cat(torch.cat([x,m], dim =-1))
        x = F.silu(x)
        x = self.agg_3(x, edge_index)
        x = self.out(x)

        return x, m