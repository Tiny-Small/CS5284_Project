import torch
import torch_scatter
import torch.nn.functional as F
from collections import namedtuple
from sklearn.metrics import precision_score, recall_score, f1_score

# Named tuples for flexible outputs
Output = namedtuple("Output", ["node_embedding", "question_embedding_expanded"])
FullOutput = namedtuple("FullOutput", ["output", "threshold"])
Metrics = namedtuple("Metrics", ["accuracy", "precision", "recall", "f1", "hits_at_k", "full_accuracy"])


def threshold_based_candidates(output, threshold=0.8):
    attributes_to_check = ["node_embedding", "question_embedding_expanded"]

    if all(hasattr(output, attr) for attr in attributes_to_check):
        # Calculate cosine similarity for each node
        similarity_scores = F.cosine_similarity(output.node_embedding, output.question_embedding_expanded, dim=1)
        # Initialize a mask to identify candidates above the threshold
        candidates_mask = similarity_scores > threshold
    # else:
    #     print("Output lacks expected attributes.")

    try:
        if output.shape[1] == 1:
            candidates_mask = torch.sigmoid(output.squeeze()) > threshold
            similarity_scores = None
    except AttributeError:
        # print("Output has no 'shape' attribute.")
        pass
    except IndexError:
        # print("Output has 'shape' but insufficient dimensions.")
        pass
    return candidates_mask.int(), similarity_scores


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


def hits_at_k_per_subgraph(preds, labels, k):
    """Calculate hits@k for a single subgraph."""
    # Sort the predictions by similarity scores
    sorted_indices = preds.argsort(descending=True)
    top_k_indices = sorted_indices[:k]
    top_k_labels = labels[top_k_indices]

    # Check if there's at least one positive label in the top-k predictions
    hits_at_k = int((top_k_labels == 1).any())
    return hits_at_k


def calculate_avg_metrics(batched_subgraphs, stacked_labels, candidates_mask, equal_subgraph_weighting, k):
    # Initialize metrics lists
    batch_accuracies, batch_precisions, batch_recalls, batch_f1s, batch_hits_at_k, batch_full = [], [], [], [], [], []
    num_subgraphs = batched_subgraphs.batch.unique().size(0)

    labels_flat = stacked_labels.view(-1)
    correct_predictions = (candidates_mask == labels_flat).int()
    nodes_per_subgraph = batched_subgraphs.batch.bincount(minlength=num_subgraphs)

    correct_per_subgraph = torch_scatter.scatter_add(correct_predictions, batched_subgraphs.batch, dim=0)
    subgraph_accuracies = correct_per_subgraph / nodes_per_subgraph
    all_correct = (correct_per_subgraph == nodes_per_subgraph).float()
    batch_accuracies.extend(subgraph_accuracies.cpu().tolist())
    full_accuracies = all_correct.cpu().tolist()
    batch_full.extend(full_accuracies)

    # Compute precision, recall, and F1 per subgraph if using equal weighting
    for i in range(num_subgraphs):
        node_mask = (batched_subgraphs.batch == i)
        labels_subgraph = stacked_labels[node_mask].squeeze().cpu().numpy()
        candidates_subgraph = candidates_mask[node_mask].cpu().numpy()
        if len(set(labels_subgraph)) > 1:
            precision = precision_score(labels_subgraph, candidates_subgraph, average='binary', zero_division=0)
            recall = recall_score(labels_subgraph, candidates_subgraph, average='binary', zero_division=0)
            f1 = f1_score(labels_subgraph, candidates_subgraph, average='binary', zero_division=0)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        # Calculate hits@k for this subgraph
        hits_at_k = hits_at_k_per_subgraph(torch.tensor(candidates_subgraph), torch.tensor(labels_subgraph), k)

        # Append per-subgraph metrics to batch-level lists
        batch_precisions.append(precision)
        batch_recalls.append(recall)
        batch_f1s.append(f1)
        batch_hits_at_k.append(hits_at_k)

    # Aggregate metrics for equal_subgraph_weighting
    if equal_subgraph_weighting:
        avg_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        avg_precision = sum(batch_precisions) / len(batch_precisions)
        avg_recall = sum(batch_recalls) / len(batch_recalls)
        avg_f1 = sum(batch_f1s) / len(batch_f1s)
        avg_hits_at_k = sum(batch_hits_at_k) / len(batch_hits_at_k)
        avg_full_accuracy = sum(batch_full) / len(batch_full)

    else:
        # Calculate for the entire batch if not using equal weighting
        avg_accuracy = correct_predictions.sum().float() / labels_flat.size(0)
        avg_precision = precision_score(labels_flat.cpu().numpy(), candidates_mask.cpu().numpy(), average='binary', zero_division=0)
        avg_recall = recall_score(labels_flat.cpu().numpy(), candidates_mask.cpu().numpy(), average='binary', zero_division=0)
        avg_f1 = f1_score(labels_flat.cpu().numpy(), candidates_mask.cpu().numpy(), average='binary', zero_division=0)
        avg_hits_at_k = -999  # Hits@k not calculated at batch level; using -999 to keep it as a number
        avg_full_accuracy = -999  # Full accuracy not calculated at batch level; using -999 to keep it as a number

    return avg_accuracy, avg_precision, avg_recall, avg_f1, avg_hits_at_k, avg_full_accuracy


def adaptive_threshold_penalty(
    output, threshold, stacked_labels, batched_subgraphs, equal_subgraph_weighting,
    target_precision=0.8, target_recall=0.8, weight=0.1, k=3
):
    candidates_mask, _ = threshold_based_candidates(output, threshold)

    _, avg_precision, avg_recall, _, _, _ = calculate_avg_metrics(
        batched_subgraphs=batched_subgraphs,
        stacked_labels=stacked_labels,
        candidates_mask=candidates_mask,
        equal_subgraph_weighting=equal_subgraph_weighting,
        k=k
    )

    # Calculate penalties based on the deviation from target precision and recall
    target_precision = torch.tensor(target_precision, dtype=torch.float32)
    avg_precision = torch.tensor(avg_precision, dtype=torch.float32)
    precision_penalty = F.relu(target_precision - avg_precision) ** 2

    target_recall = torch.tensor(target_recall, dtype=torch.float32)
    avg_recall = torch.tensor(avg_recall, dtype=torch.float32)
    recall_penalty = F.relu(target_recall - avg_recall) ** 2

    # Combine penalties
    adaptive_penalty = weight * (precision_penalty + recall_penalty)
    return adaptive_penalty


def custom_loss_fn(full_output, stacked_labels, base_loss_fn, batched_subgraphs, equal_subgraph_weighting):

    # Check if full_output contains threshold (FullOutput) or not (Output)
    output = full_output.output if isinstance(full_output, FullOutput) else full_output
    threshold = full_output.threshold if isinstance(full_output, FullOutput) else None

    if equal_subgraph_weighting:
        # subgraph-level pos_weight
        num_subgraphs = batched_subgraphs.batch.unique().size(0)
        nodes_per_subgraph = batched_subgraphs.batch.bincount(minlength=num_subgraphs).float().view(-1, 1)
        pos_count = torch_scatter.scatter_add(stacked_labels, batched_subgraphs.batch, dim=0)
        pos_weight = ((nodes_per_subgraph - pos_count) / pos_count).clamp(min=1)[batched_subgraphs.batch].squeeze()

        # Calculate loss based on whether output is embedding or logits
        if isinstance(output, Output):
            target = 2 * stacked_labels.squeeze().float() - 1
            base_loss = base_loss_fn(output.node_embedding, output.question_embedding_expanded, target) * pos_weight
        else:
            pos_weight = pos_weight.view(-1,1)
            base_loss = base_loss_fn(output, stacked_labels.float()) * pos_weight

        # Sum and normalize by subgraph
        loss_per_subgraph = torch_scatter.scatter_add(base_loss, batched_subgraphs.batch, dim=0)
        loss = (loss_per_subgraph / nodes_per_subgraph).mean()
    else:
        # batch-level pos_weight
        pos_count = stacked_labels.sum()
        neg_count = stacked_labels.numel() - pos_count
        pos_weight = (neg_count / pos_count).clamp(min=1)

        # Calculate loss based on whether output is embedding or logits
        if isinstance(output, Output):
            target = 2 * stacked_labels.squeeze().float() - 1
            loss = base_loss_fn(output.node_embedding, output.question_embedding_expanded, target) * pos_weight
        else:
            loss = base_loss_fn(output, stacked_labels.float()) * pos_weight

    # Apply threshold penalty only if threshold exists (for FullOutput)
    if threshold is not None:
        # threshold_penalty = 0.1 * (threshold - 0.65) ** 2
        threshold_penalty = adaptive_threshold_penalty(output, threshold, stacked_labels.float(), batched_subgraphs, equal_subgraph_weighting)
        return loss + threshold_penalty
    return loss
