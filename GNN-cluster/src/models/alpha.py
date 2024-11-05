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
        candidates_mask = torch.sigmoid(similarity_scores) > threshold
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


def custom_loss_fn(full_output, stacked_labels, base_loss_fn, batched_subgraphs, equal_subgraph_weighting, contrastive_loss_type, margin, temperature, reduction_type):

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
        # contrastive loss
        if isinstance(output, Output):
            if contrastive_loss_type == 'margin':
                # margin-based contrastive loss
                # encourages embeddings within this margin to move closer and those outside it to move further away
                base_loss = base_loss_fn(output.question_embedding_expanded, output.node_embedding, stacked_labels.squeeze(), reduction_type, margin) * pos_weight
            
            elif contrastive_loss_type == 'MNRL':
                # MNRL contrastive loss
                # allows all batch elements to act as negative examples as well
                base_loss = base_loss_fn(output.question_embedding_expanded, output.node_embedding, reduction_type, temperature) * pos_weight

            # anything else, by default uses default
            else:
                # converts into +1 (answer) and -1 (non-answer),
                # minimize the cosine distance between "similar" pairs of embeddings while maximizing it between "dissimilar" pairs
                target = 2 * stacked_labels.squeeze().float() - 1
                base_loss = base_loss_fn(output.node_embedding, output.question_embedding_expanded, target) * pos_weight
        
        # binary classification cross entropy loss
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

# margin-based contrastive loss
def margin_contrastive(query_embeddings, embeddings, labels, reduction, margin=0.5):
    """
    Computes margin-based contrastive cosine embedding loss for multiple positives and negatives.

    For positive examples, loss is minimized when cos(x,y) is close to 1

    For negative examples, the cosine similarity should ideally be less than the specified margin,
    meaning the embeddings should be dissimilar by at least that amount.
    The margin enforces a minimum distance by penalizing cosine similarities that exceed it.

    If you set margin = 0.5, the model will push the cosine similarity below 0.5 for negative pairs
    while encouraging positive pairs to have high similarity.
    """
    # Separate positive and negative embeddings based on labels
    positive_embeddings = embeddings[labels == 1]
    positive_corresponding_query_embeddings = query_embeddings[labels == 1]
    negative_embeddings = embeddings[labels == 0]
    negative_corresponding_query_embeddings = query_embeddings[labels == 0]

    # Positive loss: Minimize distance between query and positive samples (target = 1)
    if len(positive_embeddings) > 0:
        positive_loss = F.cosine_embedding_loss(
            positive_corresponding_query_embeddings,
            positive_embeddings,
            torch.ones(len(positive_embeddings)).to(query_embeddings.device),
            reduction=reduction
        )
    else:
        positive_loss = torch.tensor(0.0, device=query_embeddings.device)

    # Negative loss: Maximize distance for query and negative samples (target = -1)
    if len(negative_embeddings) > 0:
        negative_loss = F.cosine_embedding_loss(
            negative_corresponding_query_embeddings,
            negative_embeddings,
            torch.full((len(negative_embeddings),), -1.0).to(query_embeddings.device),
            margin=margin,
            reduction=reduction
        )
    else:
        negative_loss = torch.tensor(0.0, device=query_embeddings.device)

    # Combine losses
    if reduction == "mean":
        total_loss = (positive_loss.mean() + negative_loss.mean()) / 2 # is this correct? maybe not used
    else:
        # none
        total_loss = torch.cat([positive_loss, negative_loss], dim=0)
    
    return total_loss

# MNRL contrastive loss
def mnrl_contrastive(question_embedding_expanded, node_embedding, reduction_type, temperature=0.05):
    """
    Computes a contrastive loss with implicit negatives within the batch and explicit negatives.

    This setup leverages implicit negatives in similarity_matrix,
    where each off-diagonal entry serves as a "negative" for each query-node pair.
    
    Each item in the batch acts as a negative for every other item,
    ensuring each query only considers its paired node as a positive.

    The loss function applies cross-entropy to encourage the diagonal values to be the highest
    (i.e., most similar).
    """
    # Compute cosine similarity matrix (batch_size, batch_size)
    similarity_matrix = F.cosine_similarity(question_embedding_expanded.unsqueeze(1), node_embedding.unsqueeze(0), dim=2)

    # Scale similarities by temperature
    similarity_matrix /= temperature

    # Apply softmax to get probabilities, focusing on the diagonal
    # target labels are along the diagonal
    diagonal_labels = torch.arange(similarity_matrix.size(0)).to(node_embedding.device)
    loss = F.cross_entropy(similarity_matrix, diagonal_labels, reduction=reduction_type)

    return loss

    # # Separate positive and negative node embeddings
    # positive_nodes = node_embedding[labels == 1]
    # negative_nodes = node_embedding[labels == 0]
    # positive_q_embeddings = question_embedding_expanded[labels == 1]
    # negative_q_embeddings = question_embedding_expanded[labels == 0]

    # # Implicit negative similarities for all pairs (question to node)
    # similarity_matrix = F.cosine_similarity(question_embedding_expanded.unsqueeze(1), node_embedding.unsqueeze(0), dim=2) / temperature
    # labels_for_implicit = torch.arange(similarity_matrix.size(0)).to(node_embedding.device)
    # implicit_negative_loss = F.cross_entropy(similarity_matrix, labels_for_implicit, reduction=reduction_type)

    # # Explicit positive and negative loss
    # cosine_loss_fn = CosineEmbeddingLoss(margin=margin, reduction=reduction_type)
    # positive_loss = cosine_loss_fn(positive_q_embeddings, positive_nodes, torch.ones(positive_nodes.size(0)).to(node_embedding.device))
    # negative_loss = cosine_loss_fn(negative_q_embeddings, negative_nodes, torch.full((negative_nodes.size(0),), -1.0).to(node_embedding.device))

    # # Combine the losses
    # total_loss = implicit_negative_loss + positive_loss + negative_loss
    # return total_loss