import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
# from src.models.alpha import FullOutput, Metrics, threshold_based_candidates, calculate_avg_metrics
from models.alpha import FullOutput, Metrics, threshold_based_candidates, calculate_avg_metrics

def evaluate(dataloader, model, device, equal_subgraph_weighting, threshold_value, k=1):
    model.eval()

    with torch.no_grad():
        for batched_subgraphs, question_embeddings, stacked_labels, _, _ in tqdm(dataloader, desc="Evaluation Progress", leave=True):
            batched_subgraphs = batched_subgraphs.to(device)
            question_embeddings = question_embeddings.to(device)
            stacked_labels = stacked_labels.to(device)

            full_output = model(batched_subgraphs, question_embeddings) # Forward pass

            # Check if full_output contains threshold (FullOutput) or not (Output)
            output = full_output.output if isinstance(full_output, FullOutput) else full_output
            threshold = full_output.threshold if isinstance(full_output, FullOutput) else threshold_value

            candidates_mask, similarity_scores = threshold_based_candidates(output, threshold=threshold)

            accuracy, precision, recall, f1, hits_at_k, full_accuracy = \
                calculate_avg_metrics(
                    batched_subgraphs=batched_subgraphs,
                    stacked_labels=stacked_labels,
                    candidates_mask=candidates_mask,
                    equal_subgraph_weighting=equal_subgraph_weighting,
                    k=k
                )

    return Metrics(accuracy=accuracy,
                   precision=precision,
                   recall=recall,
                   f1=f1,
                   hits_at_k=hits_at_k,
                   full_accuracy=full_accuracy)
