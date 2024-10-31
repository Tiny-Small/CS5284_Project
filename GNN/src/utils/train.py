import torch
import os


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for (
        batched_subgraphs,
        question_embeddings,
        stacked_labels,
        node_maps,
        original_labels,
    ) in data_loader:
        optimizer.zero_grad()

        # Move tensors to the appropriate device
        batched_subgraphs = batched_subgraphs.to(device)
        question_embeddings = question_embeddings.to(device)
        stacked_labels = stacked_labels.to(device)

        # Calculate pos_weight based on the number of positive vs negative labels
        pos_weight = torch.tensor(
            [len(stacked_labels) / stacked_labels.sum()], device=device
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Forward pass
        out = model(batched_subgraphs, question_embeddings)  # Model's forward pass

        # Ensure that `stacked_labels` matches the dimensions of `out`
        stacked_labels = stacked_labels.view(-1)  # Flatten labels to match output shape

        # Ensure the shapes match
        assert (
            out.shape == stacked_labels.shape
        ), f"Shape mismatch: {out.shape} vs {stacked_labels.shape}"

        # Calculate loss
        batch_loss = loss_fn(out, stacked_labels.float())

        # Backward pass and optimization step
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, config, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        os.path.join(save_dir, f"model_epoch_{epoch}.pth"),
    )


def train(model, data_loader, optimizer, device):
    epoch_loss = train_one_epoch(model, data_loader, optimizer, device)
    print(epoch_loss)
    return epoch_loss
