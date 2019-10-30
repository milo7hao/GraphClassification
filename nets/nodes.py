from typing import Dict

import dgl
import torch
import torch.nn as nn


class NodeApplyModule(nn.Module):

    def __init__(
        self, in_feats: int, out_feats: int, activation: nn.functional
    ) -> None:

        """Update the node feature h_v with ReLU(Wh_v + b).

        Args:
            in_feats (int): Input node feature size.
            out_feats (int): Output node feature size.
            activation (nn.functional): Activation function to apply
                the end of process.
        """
        super(NodeApplyModule, self).__init__()

        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node: dgl.DGLGraph.nodes) -> Dict[str, torch.Tensor]:
        """NodeApplyModule forward propagate method.

        Args:
            node (dgl.DGLGraph.nodes): Node for updating its feature h_v.

        Returns:
            Dict[str, torch.Tensor]: Dict whose key is hidden vector name
                and value is actual hidden vector (Tensor).
        """
        h = self.linear(node.data['h'])
        h = self.activation(h)

        return {'h': h}
