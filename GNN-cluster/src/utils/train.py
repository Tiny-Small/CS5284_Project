import torch
import os
import torch_scatter
from tqdm import tqdm
import torch.nn as nn
from src.models.threshold_model import ThresholdedModel, custom_loss_fn

def train_one_epoch(model, train_loader, optimizer, device, equal_subgraph_weighting):
    model.train()
    total_loss = 0
    reduction_type = 'none' if equal_subgraph_weighting else 'mean'
    base_loss_fn = nn.BCEWithLogitsLoss(reduction=reduction_type)

    for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(train_loader, desc="Training Progress", leave=True):
        batched_subgraphs = batched_subgraphs.to(device)
        question_embeddings = question_embeddings.to(device)
        stacked_labels = stacked_labels.to(device)

        optimizer.zero_grad()

        # Forward pass, handling ThresholdedModel differently
        if isinstance(model, ThresholdedModel):
            logits, threshold = model(batched_subgraphs, question_embeddings)  # Model returns logits and threshold
        else:
            logits = model(batched_subgraphs, question_embeddings)  # Regular model returns logits only
            # threshold = 0.5


        if equal_subgraph_weighting:
            # Calculate subgraph-level pos_weight
            num_subgraphs = len(batched_subgraphs.batch.unique())
            nodes_per_subgraph = batched_subgraphs.batch.bincount(minlength=num_subgraphs).float().view(-1, 1)
            pos_count = torch_scatter.scatter_add(stacked_labels, batched_subgraphs.batch, dim=0)
            pos_weight = (nodes_per_subgraph / pos_count).clamp(min=1)

            # Apply the base loss function with per-node weighting
            if isinstance(model, ThresholdedModel):
                loss = custom_loss_fn(logits, stacked_labels.float(), threshold, base_loss_fn, pos_weight[batched_subgraphs.batch])
            else:
                loss = base_loss_fn(logits, stacked_labels.float()) * pos_weight[batched_subgraphs.batch]
            # Sum the losses for all nodes in each subgraph and normalize
            loss_per_subgraph = torch_scatter.scatter_add(loss, batched_subgraphs.batch, dim=0)
            loss = (loss_per_subgraph / nodes_per_subgraph).mean()

        else:
            # Calculate per-batch 'pos_weight'
            pos_count = stacked_labels.sum(dim=0)
            neg_count = stacked_labels.numel() - pos_count
            pos_weight = (neg_count / pos_count).clamp(min=1)  # Avoid division by zero
            # Apply the base_loss_fn with batch pos_weight
            if isinstance(model, ThresholdedModel):
                loss = custom_loss_fn(logits, stacked_labels.float(), threshold, base_loss_fn, pos_weight)
            else:
                loss = base_loss_fn(logits, stacked_labels.float()) * pos_weight

        loss.backward()
        optimizer.step()

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
