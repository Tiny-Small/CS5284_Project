import torch
import torch_scatter
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from src.models.threshold_model import ThresholdedModel, custom_loss_fn

def hits_at_k_kgqa_scatter(preds, labels, subgraph_batch, k=1):
    """
    Calculate the hits@k metric for each subgraph using torch_scatter.

    Args:
        preds (Tensor): The predictions tensor, assumed to be of shape [num_nodes].
        labels (Tensor): The ground truth labels tensor, assumed to be of shape [num_nodes].
        subgraph_batch (Tensor): Tensor that indicates which subgraph each node belongs to,
                                 same length as preds and labels.
        k (int): The value of k for the hits@k calculation.

    Returns:
        List[float]: List of hits@k accuracies for each subgraph in the batch.
    """

    # For each node in the batch, check if it's a correct prediction
    correct_predictions = (preds == labels).int()

    # Rank predictions (probabilities) within each subgraph
    _, indices = torch.topk(preds, k, dim=0)  # Take top-k predictions per node

    # Flatten predictions to compute hits@k for each subgraph
    top_k_predictions = torch.zeros_like(preds)
    top_k_predictions[indices] = 1  # Mark top-k predictions

    # Calculate hits@k for each subgraph
    hits_per_subgraph = torch_scatter.scatter_add(top_k_predictions * correct_predictions, subgraph_batch, dim=0)
    labels_per_subgraph = torch_scatter.scatter_add(labels, subgraph_batch, dim=0)

    # Calculate hits@k accuracies
    hits_at_k_accuracies = (hits_per_subgraph / labels_per_subgraph).cpu().tolist()

    return hits_at_k_accuracies


def evaluate(dataloader, model, device, equal_subgraph_weighting, k=1):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    batch_accuracies = []
    batch_precisions = []
    batch_recalls = []
    batch_f1s = []
    batch_hits_at_k = []
    batch_full = []

    with torch.no_grad():
        for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(dataloader, desc="Evaluation Progress", leave=True):
            batched_subgraphs = batched_subgraphs.to(device)
            question_embeddings = question_embeddings.to(device)
            stacked_labels = stacked_labels.to(device)

            # Forward pass, handling ThresholdedModel differently
            if isinstance(model, ThresholdedModel):
                output, threshold = model(batched_subgraphs, question_embeddings)  # Model returns logits and threshold
            else:
                output = model(batched_subgraphs, question_embeddings)  # Regular model returns logits only
                threshold = 0.5

            preds = (torch.sigmoid(output) > threshold).int()

            # Flatten predictions and labels
            predicted_flat = preds.view(-1)
            labels_flat = stacked_labels.view(-1)

            # Collect for unweighted metrics
            all_preds.extend(predicted_flat.tolist())
            all_labels.extend(labels_flat.tolist())

            # Calculate accuracy and metrics
            if equal_subgraph_weighting:
                # metrics per subgraph
                correct_predictions = (predicted_flat == labels_flat).int()
                num_subgraphs = len(batched_subgraphs.batch.unique())
                nodes_per_subgraph = batched_subgraphs.batch.bincount(minlength=num_subgraphs)

                correct_per_subgraph = torch_scatter.scatter_add(correct_predictions, batched_subgraphs.batch, dim=0) # Sum correct predictions per subgraph
                subgraph_accuracies = correct_per_subgraph / nodes_per_subgraph # Calculate accuracy per subgraph

                all_correct = (correct_per_subgraph == nodes_per_subgraph).float() # Full calculation as per paper
                full_accuracies = all_correct / nodes_per_subgraph

                batch_accuracies.extend(subgraph_accuracies.cpu().tolist()) # Append accuracies for each subgraph in the batch
                batch_full.extend(full_accuracies.cpu().tolist()) # Append accuracies for each subgraph in the batch

                # Compute precision, recall, and F1-score per subgraph
                for i in range(num_subgraphs):
                    node_mask = (batched_subgraphs.batch == i)
                    # input(f"node_mask:{node_mask}")
                    labels_subgraph = labels_flat[node_mask].cpu().numpy()
                    # input(f"sum(labels_subgraph):{sum(labels_subgraph)} | len(set(labels_subgraph)):{len(set(labels_subgraph))}")
                    predicted_subgraph = predicted_flat[node_mask].cpu().numpy()
                    # input(f"sum(predicted_subgraph):{sum(predicted_subgraph)}")

                    # input(f"len(set(labels_subgraph)) > 1:{len(set(labels_subgraph)) > 1}")
                    if len(set(labels_subgraph)) > 1:
                        precision = precision_score(labels_subgraph, predicted_subgraph, average='binary', zero_division=0)
                        recall = recall_score(labels_subgraph, predicted_subgraph, average='binary', zero_division=0)
                        f1 = f1_score(labels_subgraph, predicted_subgraph, average='binary', zero_division=0)
                        # input(f"precision:{precision}, recall:{recall}, f1:{f1}")
                    else:
                        precision, recall, f1 = 0.0, 0.0, 0.0

                    # input(f"precision:{precision}, recall:{recall}, f1:{f1}")
                    batch_precisions.append(precision)
                    batch_recalls.append(recall)
                    batch_f1s.append(f1)

                # Calculate hits@k metric per subgraph
                hits_at_k_scores = hits_at_k_kgqa_scatter(preds=predicted_flat, labels=labels_flat, subgraph_batch=batched_subgraphs.batch, k=k)
                batch_hits_at_k.extend(hits_at_k_scores)  # Add hits@k per subgraph to batch results

            else:
                # Calculate the number of correct predictions
                correct += (predicted_flat == labels_flat).sum().item()
                total += labels_flat.numel()


    # Calculate accuracy, precision, recall, and F1 score
    if equal_subgraph_weighting:
        accuracy = sum(batch_accuracies) / len(batch_accuracies)
        precision = sum(batch_precisions) / len(batch_precisions)
        recall = sum(batch_recalls) / len(batch_recalls)
        f1 = sum(batch_f1s) / len(batch_f1s)
        hits_at_k = sum(batch_hits_at_k) / len(batch_hits_at_k)  # Average hits@k score
        full_accuracy = sum(batch_full) / len(batch_full)
    else:
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        hits_at_k = None # Not calculated in this mode
        full_accuracy = None # Not calculated in this mode

    return accuracy, precision, recall, f1, hits_at_k, full_accuracy
