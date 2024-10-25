import torch
import os

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batched_subgraphs, question_embeddings, stacked_labels, _, _ in train_loader:
        batched_subgraphs = batched_subgraphs.to(device)
        stacked_labels = stacked_labels.to(device)
        
        optimizer.zero_grad()
        output = model(batched_subgraphs)
        loss = loss_fn(output, stacked_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def save_checkpoint(model, optimizer, epoch, config, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

def train(model, train_loader, optimizer, loss_fn, config, device):
    for epoch in range(config['num_epochs']):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
        
        # Save model checkpoint
        save_checkpoint(model, optimizer, epoch+1, config)
