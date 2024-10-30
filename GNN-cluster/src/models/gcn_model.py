import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv

class GCNModel(nn.Module):
    def __init__(self, in_channels, question_embedding_dim, hidden_channels, out_channels, num_GCNCov, PROC_QN_EMBED_DIM, PROC_X_DIM):
        super(GCNModel, self).__init__()
        self.num_GCNCov = num_GCNCov
        # PROC_QN_EMBED_DIM: Reduced question embedding dimension
        # PROC_X_DIM: Reduced node embedding dimension

        # Initialize GCN layers with the specified number
        self.gcn_layers = nn.ModuleList()
        for i in range(num_GCNCov):
            if i == 0:
                self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
            elif i == num_GCNCov - 1:
                self.gcn_layers.append(GCNConv(hidden_channels, PROC_X_DIM)) # Last layer to reduce to PROC_X_DIM
            else:
                self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels)) # Intermediate layers

        # Fully connected layer for reducing question embeddings
        self.fc0 = nn.Linear(question_embedding_dim, PROC_QN_EMBED_DIM)

        # Fully connected layer applied to each node embedding (after concatenating with question embedding)
        self.fc1 = nn.Linear(PROC_X_DIM + PROC_QN_EMBED_DIM, hidden_channels)  # FCL for each node
        self.fc2 = nn.Linear(hidden_channels, out_channels)  # Final output (binary classification per node)

    def forward(self, data, question_embedding):
        # Graph propagation through GCN layers
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = F.relu(x)  # Apply ReLU after each GCN layer

        # Reduce the question embedding
        question_embedding = F.relu(self.fc0(question_embedding))  # Shape: (batch_size, PROC_QN_EMBED_DIM)

        # Broadcast question embedding to match each node in the batch
        question_embedding_expanded = question_embedding[batch]  # Shape: (num_nodes_total, PROC_QN_EMBED_DIM)

        # Concatenate node embeddings with question embeddings
        combined = torch.cat([x, question_embedding_expanded], dim=1)  # Shape: (num_nodes_total, PROC_X_DIM + PROC_QN_EMBED_DIM)

        # Apply FCL node-wise (same FCL for each node, which outputs per node predictions)
        x = F.relu(self.fc1(combined))  # Shape: (num_nodes_total, 8)
        x = self.fc2(x)                 # Shape: (num_nodes_total, out_channels)

        return x
