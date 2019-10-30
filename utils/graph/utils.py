import dgl
from typing import List, Tuple
import torch


def collate(
    samples: List[Tuple[dgl.DGLGraph, str]]
) -> Tuple[dgl.BatchedDGLGraph, torch.Tensor]:

    """To form a mini-batch from a given list of graph and label pairs.

    Args:
        samples (List[Tuple[dgl.DGLGraph, str]]): A list of pairs,
            (graph, label).

    Returns:
        Tuple[dgl.BatchedDGLGraph, torch.Tensor]: A pair of batched graph
            and tensor-transformed labels.
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels)
