import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        question_embedding_dim,
    ):
        super(GCNModel, self).__init__()

        # Define GCN layers with the first layer concatenating question embeddings
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels + question_embedding_dim, hidden_channels)
        )

        # Define hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer should output one value per node for binary classification
        self.convs.append(GCNConv(hidden_channels, 1))

    def forward(self, batched_subgraphs, question_embeddings):
        # Assuming question_embeddings shape: [batch_size, embedding_dim]
        question_emb_expanded = []

        # Expand and concatenate question embedding to each node in each subgraph
        for i, subgraph in enumerate(batched_subgraphs.to_data_list()):
            expanded_q_emb = (
                question_embeddings[i].unsqueeze(0).expand(subgraph.x.size(0), -1)
            )
            subgraph.x = torch.cat((subgraph.x, expanded_q_emb), dim=1)
            question_emb_expanded.append(subgraph.x)

        # Concatenate all node features across subgraphs into a single batch
        batched_subgraphs.x = torch.cat(question_emb_expanded, dim=0)
        x, edge_index = batched_subgraphs.x, batched_subgraphs.edge_index

        # Pass through GCN layers
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))  # Apply ReLU to all layers except the last

        # Final GCN layer without activation (output is logits for BCEWithLogitsLoss)
        x = self.convs[-1](x, edge_index)
        # Return only the output logits tensor (it should now be [N, 1])
        return x.squeeze(-1)  # Ensure that this returns [N]
