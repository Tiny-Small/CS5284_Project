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
            output = model(batched_subgraphs, question_embeddings).cpu()

            # Assume a binary classification task for node-level predictions
            preds = (torch.sigmoid(output) > 0.5).int()

            # Flatten tensors to match the dimensions correctly if necessary
            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(stacked_labels.view(-1).tolist())

            # Calculate the number of correct predictions
            correct += (preds == stacked_labels).sum().item()
            total += stacked_labels.numel()  # total number of elements in labels

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall, f1
