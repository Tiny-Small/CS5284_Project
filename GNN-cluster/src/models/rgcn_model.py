# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
# from src.models.alpha import Output
from models.alpha import Output

class RGCNModel(torch.nn.Module):
    def __init__(self, node_dim, question_dim, hidden_dim, num_relations, output_dim, num_rgcn, reduced_qn_dim, reduced_node_dim, output_embedding):
        super(RGCNModel, self).__init__()
        self.output_embedding = output_embedding
        if self.output_embedding:
            output_dim = question_dim # Set output_dim to question_dim if outputting embeddings

        self.fc_reduce_qn = nn.Linear(question_dim, reduced_qn_dim)  # FCL to reduce question embedding
        self.reduce_node_dim_layer = RGCNConv(node_dim, reduced_node_dim, num_relations)  # RGCN to reduce node dim

        self.input_layer = RGCNConv(reduced_node_dim + reduced_qn_dim, hidden_dim, num_relations) # Initial layer combining reduced node and question dime
        self.rgcn_layers = nn.ModuleList([RGCNConv(hidden_dim, hidden_dim, num_relations) for _ in range(num_rgcn - 2)]) # Dynamic RGCN layers
        self.output_layer = RGCNConv(hidden_dim, output_dim, num_relations) # Output layer

    def forward(self, batched_subgraphs, question_embedding):
        x, edge_index, batch = batched_subgraphs.x, batched_subgraphs.edge_index, batched_subgraphs.batch
        edge_attr = batched_subgraphs.edge_attr

        initial_question_embedding_expanded = question_embedding[batch]
        question_embedding = F.elu(self.fc_reduce_qn(question_embedding)) # Reduce question embedding size

        x = F.elu(self.reduce_node_dim_layer(x, edge_index, edge_attr))
        question_embedding_expanded = question_embedding[batch]
        combined = torch.cat([x, question_embedding_expanded], dim=1)

        # Initial layer
        x = F.elu(self.input_layer(combined, edge_index, edge_attr))

        # Dynamic RGCN layers
        for rgcn_layer in self.rgcn_layers:
            x = F.elu(rgcn_layer(x, edge_index, edge_attr))

        # Output layer
        x = self.output_layer(x, edge_index, edge_attr)

        # Return logits or embeddings based on the flag
        if self.output_embedding:
            return Output(node_embedding=x, question_embedding_expanded=initial_question_embedding_expanded)
        else:
            return x
