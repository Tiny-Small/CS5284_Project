# gcn model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# from src.models.alpha import Output
from models.alpha import Output

class GCNModel(nn.Module):
    def __init__(self, in_channels, question_embedding_dim, hidden_channels, out_channels, num_GCNCov,
                 PROC_QN_EMBED_DIM, PROC_X_DIM, output_embedding, use_residuals=True):
        super(GCNModel, self).__init__()
        self.num_GCNCov = num_GCNCov
        self.output_embedding = output_embedding
        self.use_residuals = use_residuals  # Flag to enable residual connections

        if self.output_embedding:
            out_channels = question_embedding_dim  # Set output_dim to question_dim if outputting embeddings

        # Layers
        self.fc_reduce_qn = nn.Linear(question_embedding_dim, PROC_QN_EMBED_DIM) # Initial question embedding processing: Fully connected layer to reduce question embedding
        self.reduce_node_dim_layer = GCNConv(in_channels, PROC_X_DIM) # Reduce node dimensions: First GCN layer to reduce node dimension
        self.input_layer = GCNConv(PROC_X_DIM + PROC_QN_EMBED_DIM, hidden_channels) # Initial layer combining reduced node and question dimensions
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_GCNCov - 2)]) # Dynamic GCN layers
        self.output_layer = GCNConv(hidden_channels, out_channels) # Output layer

    def forward(self, data, question_embedding):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial question embedding processing
        initial_question_embedding_expanded = question_embedding[batch]
        question_embedding = F.elu(self.fc_reduce_qn(question_embedding))

        # Reduce node dimension
        x = F.elu(self.reduce_node_dim_layer(x, edge_index))
        question_embedding_expanded = question_embedding[batch]
        combined = torch.cat([x, question_embedding_expanded], dim=1)

        # Initial layer
        x = F.elu(self.input_layer(combined, edge_index))

        # Track intermediate outputs for residual connections
        residual = x  # Initialize the first residual connection

        # Dynamic GCN layers with residual connections
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = F.elu(gcn_layer(x, edge_index))

            # Add residual connection every two layers
            if self.use_residuals and (i + 1) % 2 == 0:
                x = x + residual  # Apply residual connection
                residual = x      # Update residual to current x

        # Output layer
        x = self.output_layer(x, edge_index)

        # Return logits or embeddings based on the flag
        if self.output_embedding:
            return Output(node_embedding=x, question_embedding_expanded=initial_question_embedding_expanded)
        else:
            return x

# class GCNModel(nn.Module):
#     def __init__(self, in_channels, question_embedding_dim, hidden_channels, out_channels, num_GCNCov,
#                  PROC_QN_EMBED_DIM, PROC_X_DIM, output_embedding, use_residuals=True):
#         super(GCNModel, self).__init__()
#         self.num_GCNCov = num_GCNCov
#         self.output_embedding = output_embedding
#         self.use_residuals = use_residuals  # Flag to enable residual connections

#         if self.output_embedding:
#             out_channels = question_embedding_dim  # Set output_dim to question_dim if outputting embeddings

#         # Initialize GCN layers with the specified number
#         self.gcn_layers = nn.ModuleList()
#         for i in range(num_GCNCov):
#             if i == 0:
#                 self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
#             elif i == num_GCNCov - 1:
#                 self.gcn_layers.append(GCNConv(hidden_channels, PROC_X_DIM))  # Last layer to reduce to PROC_X_DIM
#             else:
#                 self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))  # Intermediate layers

#         # Fully connected layers for reducing question embeddings and final outputs
#         self.fc0 = nn.Linear(question_embedding_dim, PROC_QN_EMBED_DIM)
#         self.fc1 = nn.Linear(PROC_X_DIM + PROC_QN_EMBED_DIM, hidden_channels)
#         self.fc2 = nn.Linear(hidden_channels, out_channels)

#     def forward(self, data, question_embedding):
#         # Graph propagation through GCN layers
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         initial_question_embedding_expanded = question_embedding[batch]

#         # Residual connection tracking
#         residual = None

#         for i, gcn_layer in enumerate(self.gcn_layers):
#             x = gcn_layer(x, edge_index)
#             x = F.elu(x)  # Activation after each GCN layer

#             # Add residual connection every two layers
#             if self.use_residuals and (i + 1) % 2 == 0:
#                 if residual is not None:
#                     x = x + residual  # Apply residual connection
#                 residual = x  # Update residual to current x for the next pair of layers

#         # Reduce the question embedding
#         question_embedding = F.elu(self.fc0(question_embedding))  # Shape: (batch_size, PROC_QN_EMBED_DIM)

#         # Broadcast question embedding to match each node in the batch
#         question_embedding_expanded = question_embedding[batch]  # Shape: (num_nodes_total, PROC_QN_EMBED_DIM)

#         # Concatenate node embeddings with question embeddings
#         combined = torch.cat([x, question_embedding_expanded], dim=1)  # Shape: (num_nodes_total, PROC_X_DIM + PROC_QN_EMBED_DIM)

#         # Apply fully connected layers for node-wise predictions
#         x = F.elu(self.fc1(combined))  # Shape: (num_nodes_total, hidden_channels)
#         x = self.fc2(x)                 # Shape: (num_nodes_total, out_channels)

#         # Return logits or embeddings based on the flag
#         if self.output_embedding:
#             return Output(node_embedding=x, question_embedding_expanded=initial_question_embedding_expanded)
#         else:
#             return x
