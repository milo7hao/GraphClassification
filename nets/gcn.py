import dgl
import torch
import torch.nn as nn
import dgl.function as fn

from nets.nodes import NodeApplyModule
from utils.graph import reduce

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


class GraphConvolutionalNetwork(nn.Module):

    def __init__(
        self, in_feats: int, out_feats: int, activation: nn.functional
    ) -> None:

        """Graph Convolutional Network.

        Args:
            in_feats (int): Input node feature size.
            out_feats (int): Output node feature size.
            activation: (nn.functional): Activation function to apply
                the end of process.
        """
        super(GraphConvolutionalNetwork, self).__init__()

        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(
        self, graph: dgl.DGLGraph, feature: torch.Tensor
    ) -> torch.Tensor:

        """GraphConvolutionalNetwork forward propagate method.

        Returns:
            torch.Tensor: Extracted graph representations.
        """
        # Initialize the node features with h.
        # ndata returns the data view of all the nodes.
        graph.ndata['h'] = feature

        # Send messages through all edges and update all nodes.
        # Additionally, apply a function `reduce`, which takes an average
        # over all neighbor node features, to update the node features
        # after receive.
        graph.update_all(msg, reduce)

        # Apply node func, i.e. update the node feature h_v by NodeApplyModule.
        graph.apply_nodes(func=self.apply_mod)

        # Return the end of hidden value.
        return graph.ndata.pop('h')
