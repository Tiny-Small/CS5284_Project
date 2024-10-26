import torch
import pandas as pd
from torch_geometric.data import Batch

def collate_fn(batch):
    """
    DataLoader expects each batch to contain tensors or arrays, but torch_geometric.data.Data objects need to be batched in a special way.
    """
    subgraphs, question_embeddings, labels, node_maps = zip(*batch)

    # Batch the subgraphs
    batched_subgraphs = Batch.from_data_list(subgraphs)

    # Stack the question embeddings and labels
    question_embeddings = torch.stack(question_embeddings)

    # Concatenate labels and reshape to (N, 1) where N is the total number of nodes in the batch
    stacked_labels = torch.cat(labels).unsqueeze(1)

    return batched_subgraphs, question_embeddings, stacked_labels, node_maps, list(labels)
