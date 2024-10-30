import torch
import torch.nn as nn

class ThresholdedModel(nn.Module):
    def __init__(self, model):
        super(ThresholdedModel, self).__init__()
        self.model = model  # Your existing GNN model
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Initialize learnable threshold

    def forward(self, data, question_embedding):
        logits = self.model(data, question_embedding)
        return logits, self.threshold  # Return both logits and threshold

# Custom loss that includes threshold
def custom_loss_fn(logits, labels, threshold, loss_fn, pos_weight):
    # Base loss (BCE)
    base_loss = loss_fn(logits, labels) * pos_weight

    # mode ratio of pos to neg across first 5k subgraphs: 5.2e-05
    threshold_penalty = 0.1 * (threshold - 0.65) ** 2
    return base_loss + threshold_penalty
