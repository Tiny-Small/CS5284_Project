import torch
import pandas as pd
from torch_geometric.data import Batch, DataLoader
from src.datasets.kgqa_dataset import KGQADataset


def collate_fn(batch):
    """
    DataLoader expects each batch to contain tensors or arrays, but torch_geometric.data.Data objects need to be batched in a special way.
    """
    subgraphs, question_embeddings, labels, node_maps, answer_type = zip(*batch)

    # Batch the subgraphs
    batched_subgraphs = Batch.from_data_list(subgraphs)

    # Stack the question embeddings and labels
    question_embeddings = torch.stack(question_embeddings)

    # Concatenate labels and reshape to (N, 1) where N is the total number of nodes in the batch
    stacked_labels = torch.cat(labels).unsqueeze(1)

    # Return only the items that are needed in training
    return (
        batched_subgraphs,
        question_embeddings,
        stacked_labels,
        node_maps,
        answer_type,
    )


def data_loader(dataset, encoding_model, config):

    dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config["node_embed"],
        path_to_idxes=config["idxes"],
        path_to_qa=config[f"{dataset}_qa_data"],
        path_to_ans_types=config[f"{dataset}_ans_types"],
        train=True,
        k=config["num_hops"],
    )

    sub_train_data = torch.utils.data.Subset(
        dataset, list(range(config[dataset]["start_idx"], config[dataset]["end_idx"]))
    )

    data_loader = DataLoader(
        sub_train_data,
        batch_size=config[dataset]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
    )

    return data_loader
