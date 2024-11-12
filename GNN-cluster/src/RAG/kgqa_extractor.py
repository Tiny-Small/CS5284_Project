import torch
from tqdm import tqdm
from src.models.alpha import threshold_based_candidates


def extract_subgraph_qemb(dataloader, model, device, equal_subgraph_weighting, threshold_value, save_all_path, save_emb_path):
    model.eval()
    
    all_batched_subgraphs = []
    all_question_embeddings = []
    all_candidates_masks = []
    all_similarity_scores = []
    all_node_maps = [] 
    all_labels = [] 

    with torch.no_grad():
        for batched_subgraphs, question_embeddings, stacked_labels, node_maps, labels in tqdm(dataloader, desc="Extracting subgraph", leave=True):
            # Move tensors to the specified device
            batched_subgraphs = batched_subgraphs.to(device)
            question_embeddings = question_embeddings.to(device)
            stacked_labels = stacked_labels.to(device)

            # Perform forward pass
            full_output = model(batched_subgraphs, question_embeddings)
            output = full_output.output if hasattr(full_output, 'output') else full_output
            threshold = full_output.threshold if hasattr(full_output, 'threshold') else threshold_value

            # Determine candidate nodes based on similarity threshold
            candidates_mask, similarity_score = threshold_based_candidates(output, threshold=threshold)  # Now expecting a single similarity score per batch

            # Save batched data to lists (detaching to avoid memory leaks)
            all_batched_subgraphs.append(batched_subgraphs.x.detach().cpu())
            all_question_embeddings.append(question_embeddings.detach().cpu())
            all_candidates_masks.append(candidates_mask.detach().cpu())
            all_node_maps.extend(node_maps) 
            all_labels.extend(labels) 
            if similarity_score is not None:
                all_similarity_scores.append(float(similarity_score))
                
    # Concatenate all batched data along the 0-axis (vertically)
    all_batched_subgraphs = torch.cat(all_batched_subgraphs, dim=0)
    all_question_embeddings = torch.cat(all_question_embeddings, dim=0)
    all_candidates_masks = torch.cat(all_candidates_masks, dim=0)

    # Saving processed data to files
    save_subg_qemb_file(all_batched_subgraphs, all_question_embeddings, file_path=save_emb_path)
    save_all_to_file(
        all_batched_subgraphs,
        all_question_embeddings,
        all_candidates_masks,
        all_similarity_scores,
        all_node_maps,
        all_labels,
        file_path=save_all_path
    )
def save_all_to_file(batched_subgraphs, question_embeddings, candidates_mask, similarity_scores, node_map, labels, file_path):
    
    data = {
        "batched_subgraphs": batched_subgraphs,
        "question_embeddings" : question_embeddings,
        "candidates_masks": candidates_mask.tolist(),
        "similarity_scores": similarity_scores.tolist() if similarity_scores is not None else None,
        "node_maps" : node_map,
        "labels" : labels
    }
    
    torch.save(data, file_path)


def save_subg_qemb_file(batched_subgraphs, question_embeddings, file_path):
    
    data = {
        "batched_subgraphs": batched_subgraphs,
        "question_embeddings" : question_embeddings,
    }
    
    torch.save(data, file_path)


def load_all_metadata(file_path):
    # Load the data from the saved file
    saved_data = torch.load(file_path)
    
    # Extract each component from the dictionary
    batched_subgraphs = saved_data["batched_subgraphs"]
    question_embeddings = saved_data["question_embeddings"]
    candidates_masks = saved_data["candidates_masks"]
    similarity_scores = saved_data.get("similarity_scores", None)
    node_maps = saved_data["node_maps"]
    labels = saved_data["labels"]
    
    return batched_subgraphs, question_embeddings, candidates_masks, similarity_scores, node_maps, labels

def load_subgraph_data(file_path):
    # Load the data from the saved file
    saved_data = torch.load(file_path)
    
    # Extract each component from the dictionary
    batched_subgraphs = saved_data["batched_subgraphs"]
    question_embeddings = saved_data["question_embeddings"]
    
    return batched_subgraphs, question_embeddings
