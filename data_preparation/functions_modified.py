import json, re, torch
import pandas as pd
import torch_geometric
from torch_geometric.utils import k_hop_subgraph
import pandas as pd
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import random

"""
# sample code how to use
from functions import *
from torch.utils.data import DataLoader

path_to_node_embed = '../Datasets/MetaQA_dataset/processed/node2vec _embeddings/ud_node2vec_embeddings.txt'
path_to_idxes = '../Datasets/MetaQA_dataset/processed/idxes.json'
path_to_qa = '../Datasets/MetaQA_dataset/vanilla 3-hop/qa_train.txt'
data = KGQADataset(path_to_node_embed, path_to_idxes, path_to_qa)

dataloader = DataLoader(data, batch_size=32, collate_fn=collate_fn, shuffle=True)

for batched_subgraphs, question_embeddings, labels in dataloader:
    break

print(question_embeddings[0].shape)
print(sum(labels[0]))
print(batched_subgraphs[0].num_nodes)
print(batched_subgraphs[0].x.shape)
"""

# SBERT (can change to others) (384 dimensional)
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

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

    # # Pad labels so they all have the same size
    # max_num_nodes = max([label.size(0) for label in labels])
    # padded_labels = [F.pad(label, (0, max_num_nodes - label.size(0)), value=0) for label in labels]

    # # Create a mask where padded entries are 0, and valid entries are 1
    # masks = [torch.ones_like(label) for label in labels]
    # padded_masks = [F.pad(mask, (0, max_num_nodes - mask.size(0)), value=0) for mask in masks]

    # # Pad labels so they all have the same size
    # max_num_nodes = max([label.size(0) for label in labels])
    # padded_labels = [F.pad(label, (0, max_num_nodes - label.size(0)), value=0) for label in labels]

    # return batched_subgraphs, question_embeddings, list(padded_labels) #list(labels)

    # return batched_subgraphs, question_embeddings, torch.stack(padded_labels), torch.stack(padded_masks)
    return batched_subgraphs, question_embeddings, stacked_labels, node_maps #list(labels)


class KGQADataset(torch.utils.data.Dataset):
    def __init__(self, path_to_node_embed, path_to_idxes, path_to_qa, k=3):
        """
        path_to_node_embed (str): Path to node embeddings (node2vec).
        path_to_idxes (str): Path to idxes extracted (entity_to_idx, edge_index, relations).
        path_to_qa (str): Path to question and answers.
        k (int): Number of hops (default is 3).
        """
        # load data object (graph)
        self.loaded_entity_to_idx, self.loaded_edge_index, self.loaded_relations = self.load_data_json(path_to_idxes)
        self.data = self.create_data_object(self.loaded_edge_index, self.loaded_relations, self.loaded_entity_to_idx)

        # load the questions and answers
        self.df = pd.read_csv(path_to_qa, sep='\t', header=None, names=['question', 'answer'])
        self.df['answer'] = self.df['answer'].apply(lambda x: x.split("|"))
        self.k = k

        # load the sentence embeddings
        self.q_embeddings = model.encode([q.replace("[", "").replace("]", "") for q in self.df['question']], batch_size=128, convert_to_tensor=True) # 32

        # load the node2vec embeddings
        self.node2vec_embeddings = self.load_node2vec_embeddings(path_to_node_embed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the question and answers from the DataFrame
        row = self.df.iloc[idx]
        question = row['question']
        answers = row['answer']

        # Step 1: Extract the entity from the question (entity marked in square brackets)
        entity = self.extract_entity_from_question(question)

        # Step 2: Get the node index corresponding to the entity
        if entity not in self.loaded_entity_to_idx:
            raise ValueError(f"Entity {entity} not found in node index.")
        entity_node = self.loaded_entity_to_idx[entity]

        # Step 3: Get the k-hop subgraph around the question entity
        subgraph_data, node_map = self.get_k_hop_subgraph(entity_node)

        # Step 4: Get the question embedding (assuming you have a function for this)
        question_embedding = self.q_embeddings[idx]

        # Step 5: Get the labels (map answer entities to their node indices)
        labels = self.get_labels(answers, node_map)

        # Step 6: Add node2vec embeddings into the subgraph data
        subgraph_data.x = self.get_node_embeddings(node_map)

        # Step 7: Append node_map as an attribute of subgraph_data
        # subgraph_data.node_map = node_map  # Adding node_map to subgraph_data

        return subgraph_data, question_embedding, labels, node_map

    def load_data_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['entity_to_idx'], data['edge_index'], data['relations']

    def create_data_object(self, edge_index, relations, entity_to_idx):
        unique_relations = list(set(relations))
        relation_mapping = {relation: index for index, relation in enumerate(unique_relations)}

        edge_index = torch.tensor(edge_index).t().contiguous()
        # Make the graph undirected by adding reverse edges
        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # comment out if want directed
        edge_attr = torch.tensor([relation_mapping[rel] for rel in relations])
        # Since we now have more edges (two for each undirected edge), we concat them
        undirected_edge_attr = torch.cat([edge_attr, edge_attr], dim=0) # comment out if want directed

        return Data(edge_index=undirected_edge_index, edge_attr=undirected_edge_attr, num_nodes=len(entity_to_idx))

    def load_node2vec_embeddings(self, file_path, embedding_dim=64):
        embeddings_dict = {}

        with open(file_path, 'r') as f:
            # Skip the first row
            next(f)

            for line in f:
                parts = line.strip().split()

                # The entity is everything before the embedding, so we use -embedding_dim
                entity = " ".join(parts[:-embedding_dim])  # Join all words before the embedding dimensions
                embedding = list(map(float, parts[-embedding_dim:]))  # Convert last parts to float

                # Store in the dictionary
                embeddings_dict[entity] = embedding

        return embeddings_dict

    def extract_entity_from_question(self, question):
        """
        Extract the entity that is enclosed in square brackets from the question.
        Example: "What city is [Paris] the capital of?" -> "Paris"
        """
        # assumes one entity of interest in each questin
        start = question.find('[') + 1
        end = question.find(']')
        if start == 0 or end == -1:
            raise ValueError(f"No entity found in the question: {question}")
        return question[start:end]

    def get_k_hop_subgraph(self, node_idx):
        """
        Get the k-hop subgraph centered around the given node index.
        node_idx (int): Index of the node representing the entity.
        """
        # Extract k-hop subgraph from the full graph
        node_idx = torch.tensor([node_idx], dtype=torch.long)
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.k,
            edge_index=self.data.edge_index,
            relabel_nodes=True
        )

        # Create a subgraph Data object
        # subgraph = Data(x=self.data.x[subset], edge_index=sub_edge_index)
        subgraph = Data(edge_index=sub_edge_index)

        # Create a mapping from original node indices to subgraph indices
        node_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(subset)}

        return subgraph, node_map

    def get_labels(self, answers, node_map):
        labels = torch.zeros(len(node_map), dtype=torch.long)
        for ans in answers:
            if ans in self.loaded_entity_to_idx:
                ans_idx = self.loaded_entity_to_idx[ans]
                if ans_idx in node_map:
                    labels[node_map[ans_idx]] = 1
        return labels

    # def get_node_embeddings(self, node_map):
    #     embeddings = [[0.0] * len(self.node2vec_embeddings[next(iter(self.node2vec_embeddings))])]*len(node_map)
    #     idx_to_entity = {v: k for k, v in self.loaded_entity_to_idx.items()}

    #     for ori, new in node_map.items():
    #         if idx_to_entity[ori] in self.node2vec_embeddings:
    #             embeddings[new] = self.node2vec_embeddings[idx_to_entity[ori]]
    #     return torch.tensor(embeddings, dtype=torch.float)

    def get_node_embeddings(self, node_map, embedding_dim=64, random_init=False):
        """
        Get node embeddings from node2vec. If node not found or random_init is True, create random embeddings.
        """
        embeddings = [[0.0] * embedding_dim] * len(node_map)  # Initialize all embeddings with zeros
        idx_to_entity = {v: k for k, v in self.loaded_entity_to_idx.items()}

        for ori, new in node_map.items():
            if random_init:
                # Randomly initialize embeddings if random_init is True
                embeddings[new] = [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
            elif idx_to_entity[ori] in self.node2vec_embeddings:
                embeddings[new] = self.node2vec_embeddings[idx_to_entity[ori]]
            # else:
                # Create random embedding for missing nodes if node2vec embedding doesn't exist
                # embeddings[new] = [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]

        return torch.tensor(embeddings, dtype=torch.float)
