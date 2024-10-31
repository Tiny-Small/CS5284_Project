import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_subgraphs, question_embeddings, stacked_labels, _, _ in dataloader:
            batched_subgraphs = batched_subgraphs.to(device)
            question_embeddings = question_embeddings.to(device)
            stacked_labels = stacked_labels.to(device)

            # Pass both the graph and question embeddings to the model
            output = model(batched_subgraphs, question_embeddings)

            # Assume a binary classification task for node-level predictions
            preds = (torch.sigmoid(output) > 0.5).int()

            # Move stacked_labels to CPU for comparison if preds is on CPU
            stacked_labels_cpu = stacked_labels.cpu()

            print("Preds shape:", preds.shape)
            print("Stacked_labels shape:", stacked_labels_cpu.shape)

            # Flatten tensors to match the dimensions correctly if necessary
            all_preds.extend(preds.view(-1).cpu().tolist())
            all_labels.extend(stacked_labels_cpu.view(-1).tolist())

            # Calculate the number of correct predictions
            correct += (preds.cpu() == stacked_labels_cpu).sum().item()
            total += stacked_labels.numel()  # total number of elements in labels

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return accuracy, precision, recall, f1
