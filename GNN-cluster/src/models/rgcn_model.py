# rgcn model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.models.alpha import Output
# from models.alpha import Output

class RGCNModel(torch.nn.Module):
    def __init__(self, node_dim, question_dim, hidden_dim, num_relations, output_dim, num_rgcn,
                 reduced_qn_dim, reduced_node_dim, output_embedding, use_residuals=True):
        super(RGCNModel, self).__init__()
        self.output_embedding = output_embedding
        self.use_residuals = use_residuals  # Flag to enable residual connections

        if self.output_embedding:
            output_dim = question_dim  # Set output_dim to question_dim if outputting embeddings

        # Layers
        self.fc_reduce_qn = nn.Linear(question_dim, reduced_qn_dim)  # FCL to reduce question embedding
        self.reduce_node_dim_layer = RGCNConv(node_dim, reduced_node_dim, num_relations)  # RGCN to reduce node dim
        self.input_layer = RGCNConv(reduced_node_dim + reduced_qn_dim, hidden_dim, num_relations)  # Initial layer
        self.rgcn_layers = nn.ModuleList([RGCNConv(hidden_dim, hidden_dim, num_relations) for _ in range(num_rgcn - 2)])  # Dynamic RGCN layers
        self.output_layer = RGCNConv(hidden_dim, output_dim, num_relations)  # Output layer

    def forward(self, batched_subgraphs, question_embedding):
        x, edge_index, batch = batched_subgraphs.x, batched_subgraphs.edge_index, batched_subgraphs.batch
        edge_attr = batched_subgraphs.edge_attr

        # Initial question embedding processing
        initial_question_embedding_expanded = question_embedding[batch]
        question_embedding = F.elu(self.fc_reduce_qn(question_embedding))

        # Reduce node dimension
        x = F.elu(self.reduce_node_dim_layer(x, edge_index, edge_attr))
        question_embedding_expanded = question_embedding[batch]
        combined = torch.cat([x, question_embedding_expanded], dim=1)

        # Initial layer
        x = F.elu(self.input_layer(combined, edge_index, edge_attr))

        # Track intermediate outputs for residual connections
        residual = x  # Initialize the first residual connection

        # Dynamic RGCN layers with residual connections
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = F.elu(rgcn_layer(x, edge_index, edge_attr))

            # Add residual every two layers
            if self.use_residuals and (i + 1) % 2 == 0:
                x = x + residual  # Apply residual connection
                residual = x      # Update residual to current x

        # Output layer
        x = self.output_layer(x, edge_index, edge_attr)

        # Return logits or embeddings based on the flag
        if self.output_embedding:
            return Output(node_embedding=x, question_embedding_expanded=initial_question_embedding_expanded)
        else:
            return x
