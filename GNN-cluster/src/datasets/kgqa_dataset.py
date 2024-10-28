import json, torch
import pandas as pd
from torch_geometric.utils import k_hop_subgraph
import pandas as pd
from torch_geometric.data import Data
import random

class KGQADataset(torch.utils.data.Dataset):
    def __init__(self, model, path_to_node_embed, path_to_idxes, path_to_qa, k=3):
        """
        Initialize without precomputed subgraphs. Computes k-hop subgraphs on-the-fly.
        """
        # Load the main graph data
        self.loaded_entity_to_idx, self.loaded_edge_index, self.loaded_relations = self.load_data_json(path_to_idxes)
        self.data = self.create_data_object(self.loaded_edge_index, self.loaded_relations, self.loaded_entity_to_idx)

        # Load question and answer data
        self.df = pd.read_csv(path_to_qa, sep='\t', header=None, names=['question', 'answer'])
        self.df['answer'] = self.df['answer'].apply(lambda x: x.split("|"))
        self.k = k

        # Load sentence embeddings
        self.q_embeddings = model.encode(
            [q.replace("[", "").replace("]", "") for q in self.df['question']],
            batch_size=128,
            convert_to_tensor=True
        )

        # Load node2vec embeddings
        self.node2vec_embeddings = self.load_node2vec_embeddings(path_to_node_embed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the question and answers from the DataFrame
        row = self.df.iloc[idx]
        question, answers = row['question'], row['answer']

        # Step 1: Extract the entity from the question (entity marked in square brackets)
        entity = self.extract_entity_from_question(question)
        if entity not in self.loaded_entity_to_idx:
            raise ValueError(f"Entity {entity} not found in node index.")
        entity_node = self.loaded_entity_to_idx[entity]

        # Step 2: Compute the k-hop subgraph around the entity dynamically
        subset_node_indices, sub_edge_index, _, _ = self.get_k_hop_subgraph(entity_node)

        # Step 3: Construct the subgraph based on these subset indices
        subgraph_data, node_map = self.construct_subgraph(subset_node_indices, sub_edge_index)

        # Step 4: Get the question embedding
        question_embedding = self.q_embeddings[idx]

        # Step 5: Get the labels
        labels = self.get_labels(answers, node_map)

        # Step 6: Add node2vec embeddings to the subgraph data
        subgraph_data.x = self.get_node_embeddings(node_map)

        return subgraph_data, question_embedding, labels, node_map

    def get_k_hop_subgraph(self, entity_node):
        """
        Compute the k-hop subgraph dynamically for a given entity node.
        """
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=entity_node,
            num_hops=self.k,
            edge_index=self.data.edge_index,
            relabel_nodes=True
        )
        return subset, sub_edge_index, mapping, edge_mask

    def construct_subgraph(self, subset_node_indices, sub_edge_index):
        """
        Construct a subgraph Data object for the given subset of nodes and edges.
        """
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset_node_indices)}

        # # Map edges to subgraph indices
        # mapped_edge_index = torch.tensor(
        #     [[node_map[src.item()], node_map[dst.item()]] for src, dst in sub_edge_index.t()],
        #     dtype=torch.long
        # ).t().contiguous()

        # # Create subgraph data object
        # subgraph_data = Data(edge_index=mapped_edge_index)
        subgraph_data = Data(edge_index=sub_edge_index) # sub_edge_index is get_k_hop_subgraph's sub_edge_index
        return subgraph_data, node_map



    def load_data_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['entity_to_idx'], data['edge_index'], data['relations']

    def load_data_pt(self, filename):
        """
        Load the list of subsets (k-hop subgraph node indices) from the .pt file.
        """
        data = torch.load(filename)
        if isinstance(data, list):
            return data  # Return the list of subsets
        else:
            raise ValueError("Expected a list of k-hop subgraph node indices.")


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

    # def get_k_hop_subgraph(self, node_idx):
    #     """
    #     Get the k-hop subgraph centered around the given node index.
    #     node_idx (int): Index of the node representing the entity.
    #     """
    #     # Extract k-hop subgraph from the full graph
    #     node_idx = torch.tensor([node_idx], dtype=torch.long)
    #     subset, sub_edge_index, _, _ = k_hop_subgraph(
    #         node_idx=node_idx,
    #         num_hops=self.k,
    #         edge_index=self.data.edge_index,
    #         relabel_nodes=True
    #     )

    #     # Create a subgraph Data object
    #     # subgraph = Data(x=self.data.x[subset], edge_index=sub_edge_index)
    #     subgraph = Data(edge_index=sub_edge_index)

    #     # Create a mapping from original node indices to subgraph indices
    #     node_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(subset)}

    #     return subgraph, node_map

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
