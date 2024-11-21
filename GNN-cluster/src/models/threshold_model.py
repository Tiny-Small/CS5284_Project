import torch
import torch.nn as nn
# from src.models.alpha import FullOutput
from models.alpha import FullOutput

class ThresholdedModel(nn.Module):
    def __init__(self, model):
        super(ThresholdedModel, self).__init__()
        self.model = model  # Existing GNN model
        self.output_embedding = getattr(model, 'output_embedding', False)  # Copy attribute if it exists
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Initialize learnable threshold

    def forward(self, data, question_embedding):
        output = self.model(data, question_embedding)
        return FullOutput(output=output, threshold=self.threshold)
