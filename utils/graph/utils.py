from typing import Dict, List, Tuple, Callable

import dgl
import torch
import dgl.function as fn


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


def reduce(nodes: dgl.DGLGraph.nodes) -> Dict[str, torch.Tensor]:

    """Take an average over all neighbor node features h_u and use it
        to overwrite the original node feature.

    Args:
        nodes (dgl.DGLGraph.nodes): Specified node to take an average
            over all neighbor node features.

    Returns:
        torch.Tensor: Accumulated hidden feature (vector).
    """
    accumulated = torch.mean(nodes.mailbox['m'], 1)

    return {'h': accumulated}


def message() -> Callable:
    """Sends a message of node feature h.

    Returns:
        Callable: Callable function sends node feature h to m.
    """
    return fn.copy_src(src='h', out='m')
