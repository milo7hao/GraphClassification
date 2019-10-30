import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.gcn import GraphConvolutionalNetwork


class Classifier(nn.Module):

    def __init__(
        self, in_dim: int, hidden_him: int, n_classes: int
    ) -> None:

        """Classifier module for GCN.

        Args:
            in_dim (int): Input dimension size.
            hidden_dim (int): Hidden dimension size.
            n_classes (int): Number of classes to classify.
        """
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GraphConvolutionalNetwork(in_dim, hidden_him, F.relu),
            GraphConvolutionalNetwork(hidden_him, hidden_him, F.relu)
        ])
        self.classify = nn.Linear(hidden_him, n_classes)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Classifier forward propagate method.

        Args:
            graph (dgl.DGLGraph): Graph to classify.

        Returns:
            torch.Tensor: Predicted classes.
        """
        # For undirected graphs, in_degree is the same as out_degree.
        h = graph.in_degrees().view(-1, 1).float()

        for conv in self.layers:
            h = conv(graph, h)

        graph.ndata['h'] = h
        h_g = dgl.mean_nodes(graph, 'h')

        return self.classify(h_g)
