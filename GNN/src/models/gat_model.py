import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=1):
        """
        Args:
            in_channels (int): Number of input features (per node).
            hidden_channels (int): Number of hidden units in each GAT layer.
            out_channels (int): Number of output features (e.g., number of classes for classification).
            num_layers (int): Number of GAT layers (default is 2).
            heads (int): Number of attention heads in each GAT layer (default is 1).
        """
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, data):
        """
        Forward pass through the GAT.
        
        Args:
            data (torch_geometric.data.Data): A graph data object containing x (node features) and edge_index (graph edges).
        
        Returns:
            torch.Tensor: Output node embeddings or class scores.
        """
        x, edge_index = data.x, data.edge_index

        # Pass through GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)  # ELU activation for GAT layers
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)

        return F.log_softmax(x, dim=1)  # Use log_softmax for classification
