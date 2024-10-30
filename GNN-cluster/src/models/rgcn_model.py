# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv

class RGCNModel(torch.nn.Module):
    def __init__(self, node_dim, question_dim, hidden_dim, num_relations, output_dim, num_rgcn, reduced_qn_dim, reduced_node_dim):
        super(RGCNModel, self).__init__()
        self.fc_reduce_qn = nn.Linear(question_dim, reduced_qn_dim)  # FCL to reduce question embedding
        self.reduce_node_dim_layer = RGCNConv(node_dim, reduced_node_dim, num_relations)  # RGCN to reduce node dim

        # Initial layer combining reduced node and question dimensions
        self.input_layer = RGCNConv(reduced_node_dim + reduced_qn_dim, hidden_dim, num_relations)

        # Dynamic RGCN layers
        self.rgcn_layers = nn.ModuleList(
            [RGCNConv(hidden_dim, hidden_dim, num_relations) for _ in range(num_rgcn - 2)]
        )

        # Output layer
        self.output_layer = RGCNConv(hidden_dim, output_dim, num_relations)

    def forward(self, batched_subgraphs, question_embedding):
        # Reduce question embedding size
        question_embedding = F.relu(self.fc_reduce_qn(question_embedding))

        x, edge_index, batch = batched_subgraphs.x, batched_subgraphs.edge_index, batched_subgraphs.batch
        edge_attr = batched_subgraphs.edge_attr

        # Step 1: Reduce node dimension if needed
        x = F.relu(self.reduce_node_dim_layer(x, edge_index, edge_attr))

        # Step 2: Broadcast question embedding to match each node in the batch
        question_embedding_expanded = question_embedding[batch]

        # Step 3: Concatenate node features and question embeddings along the feature dimension
        combined = torch.cat([x, question_embedding_expanded], dim=1)

        # Initial RGCN layer
        x = self.input_layer(combined, edge_index, edge_attr)
        x = F.relu(x)

        # Dynamic RGCN layers
        for rgcn_layer in self.rgcn_layers:
            x = rgcn_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        # Output layer
        x = self.output_layer(x, edge_index, edge_attr)

        return x
