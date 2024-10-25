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
            stacked_labels = stacked_labels.to(device)
            output = model(batched_subgraphs).cpu()

            # Calculate accuracy for each subgraph
            preds = (torch.sigmoid(output) > 0.5).int()
            all_preds.extend(preds.tolist())
            all_labels.extend(stacked_labels.tolist())
            correct += (preds == stacked_labels).sum().item()
            total += stacked_labels.size(0)

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall, f1
