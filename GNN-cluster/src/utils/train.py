import torch
import os
import torch_scatter
from tqdm import tqdm

# def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
#     model.train()
#     total_loss = 0
#     for batched_subgraphs, question_embeddings, stacked_labels, _, _ in train_loader:
#         batched_subgraphs = batched_subgraphs.to(device)
#         question_embeddings = question_embeddings.to(device)
#         stacked_labels = stacked_labels.to(device)

#         optimizer.zero_grad()
#         output = model(batched_subgraphs, question_embeddings)
#         loss = loss_fn(output, stacked_labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / len(train_loader)

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, equal_subgraph_weighting):
    model.train()
    total_loss = 0

    for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(train_loader, desc="Training Progress", leave=True):
        batched_subgraphs = batched_subgraphs.to(device)
        question_embeddings = question_embeddings.to(device)
        stacked_labels = stacked_labels.to(device)

        optimizer.zero_grad()
        output = model(batched_subgraphs, question_embeddings) # Forward pass

        loss = loss_fn(output, stacked_labels.float())

        if equal_subgraph_weighting:
            num_subgraphs = len(batched_subgraphs.batch.unique())
            nodes_per_subgraph = batched_subgraphs.batch.bincount(minlength=num_subgraphs).float()
            # loss = loss_fn(output, stacked_labels)

            # Sum the losses for all nodes in each subgraph
            loss_per_subgraph = torch_scatter.scatter_add(loss, batched_subgraphs.batch, dim=0)

            # Normalize the loss per subgraph by the number of nodes in each subgraph
            loss = (loss_per_subgraph / nodes_per_subgraph).mean()
        # else:
        #     loss = loss.mean()

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

# def train(model, train_loader, optimizer, loss_fn, config, device, equal_subgraph_weighting):
#     for epoch in range(config['train']['num_epochs']):
#         epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, equal_subgraph_weighting)
#         print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")

#         # Save model checkpoint
#         save_checkpoint(model, optimizer, epoch+1, config)

#     return epoch_loss
