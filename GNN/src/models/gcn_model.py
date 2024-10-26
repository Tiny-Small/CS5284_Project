import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        """
        Args:
            in_channels (int): Number of input features (per node).
            hidden_channels (int): Number of hidden units in each GCN layer.
            out_channels (int): Number of output features (e.g., number of classes for classification).
            num_layers (int): Number of GCN layers (default is 2).
        """
        super(GCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Optional: Add a linear layer to process question embeddings
        self.question_fc = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, data, question_embeddings):
        """
        Forward pass through the GCN.
        
        Args:
            data (torch_geometric.data.Data): A graph data object containing x (node features) and edge_index (graph edges).
            question_embeddings: 
        
        Returns:
            torch.Tensor: Output node embeddings or class scores.
        """
        x, edge_index = data.x, data.edge_index

        # Process graph with GCN layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        # Combine question embeddings with graph features
        question_embeds = self.question_fc(question_embeddings).unsqueeze(1)
        x = x + question_embeds  # Example: add question embeddings to node features

        return F.log_softmax(x, dim=1)