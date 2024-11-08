import os
from tqdm import tqdm
import torch
import torch.nn as nn

from src.models.alpha import (custom_loss_fn, margin_contrastive, mnrl_contrastive,
                          CosineEmbeddingLossWithLearnableMargin)
# from models.alpha import (custom_loss_fn, margin_contrastive, mnrl_contrastive,
#                           CosineEmbeddingLossWithLearnableMargin)

base_loss_fn = None
reduction_type = None

def train_one_epoch(model, train_loader, optimizer, device, equal_subgraph_weighting, contrastive_loss_type, margin, temperature):
    model.train()
    total_loss = 0

    global base_loss_fn, reduction_type
    if base_loss_fn is None:
        # Base loss function selection
        reduction_type = 'none' if equal_subgraph_weighting else 'mean'

        # contrastive loss
        if hasattr(model, 'output_embedding') and model.output_embedding:
            if contrastive_loss_type == 'margin':
                # margin-based contrastive loss
                # encourages embeddings within this margin to move closer and those outside it to move further away
                base_loss_fn = margin_contrastive
            elif contrastive_loss_type == 'MNRL':
                # MNRL contrastive loss
                # aside from the specified negative examples, allows all batch elements to act as negative examples as well
                base_loss_fn = mnrl_contrastive

            elif contrastive_loss_type == 'LearnableMargin':
                base_loss_fn = CosineEmbeddingLossWithLearnableMargin(margin_init=margin, reduction=reduction_type).to(device)
                optimizer.add_param_group({'params': [base_loss_fn.margin]})
            else:
                # anything else, by default uses default
                base_loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction_type)

        # binary classification cross entropy loss
        else:
            base_loss_fn = nn.BCEWithLogitsLoss(reduction=reduction_type)

    for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(train_loader, desc="Training Progress", leave=True):
        batched_subgraphs = batched_subgraphs.to(device)
        question_embeddings = question_embeddings.to(device)
        stacked_labels = stacked_labels.to(device)

        optimizer.zero_grad()
        full_output = model(batched_subgraphs, question_embeddings)

        # Custom loss with threshold handling
        loss = custom_loss_fn(full_output, stacked_labels, base_loss_fn, batched_subgraphs, equal_subgraph_weighting, contrastive_loss_type, margin, temperature, reduction_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Apply gradient clipping
        optimizer.step()
        torch.cuda.empty_cache() # Clear the cache

        total_loss += loss.item()
        # print(f"learnable margin: {base_loss_fn.margin.item()}")

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
