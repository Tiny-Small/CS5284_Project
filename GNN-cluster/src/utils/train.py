import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.threshold_model import ThresholdedModel
from src.models.alpha import custom_loss_fn
# from models.threshold_model import ThresholdedModel
# from models.alpha import custom_loss_fn

def train_one_epoch(model, train_loader, optimizer, device, equal_subgraph_weighting):
    model.train()
    total_loss = 0

    # Base loss function selection
    reduction_type = 'none' if equal_subgraph_weighting else 'mean'
    if hasattr(model, 'output_embedding') and model.output_embedding:
        base_loss_fn = lambda x, y, t: F.cosine_embedding_loss(x, y, t, reduction=reduction_type)
    else:
        base_loss_fn = nn.BCEWithLogitsLoss(reduction=reduction_type)

    for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(train_loader, desc="Training Progress", leave=True):
        batched_subgraphs = batched_subgraphs.to(device)
        question_embeddings = question_embeddings.to(device)
        stacked_labels = stacked_labels.to(device)

        optimizer.zero_grad()
        full_output = model(batched_subgraphs, question_embeddings)

        # Custom loss with threshold handling
        loss = custom_loss_fn(full_output, stacked_labels, base_loss_fn, batched_subgraphs, equal_subgraph_weighting)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache() # Clear the cache

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)

    return epoch_loss


def save_checkpoint(model, optimizer, epoch, config, save_dir='checkpoints', best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set filename based on whether this is the best model
    filename = f"model_epoch_{epoch}.pth"
    if best:
        filename = f"best_model_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, os.path.join(save_dir, filename))
