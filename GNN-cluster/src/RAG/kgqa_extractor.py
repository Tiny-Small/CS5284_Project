import torch
from tqdm import tqdm
from src.models.alpha import threshold_based_candidates
import json

def extract_subgraph_qemb(dataloader, model, device, equal_subgraph_weighting, threshold_value, save_all_path, save_emb_path):
    model.eval()
    
    all_batched_subgraphs = []
    all_question_embeddings = []
    all_candidates_masks = []
    all_similarity_scores = []  # Store per-node scores or skip if None
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

            # Calculate similarity scores and candidates mask
            candidates_mask, similarity_score = threshold_based_candidates(output, threshold=threshold)

            # Store batched data
            all_batched_subgraphs.append(batched_subgraphs.x.detach().cpu())
            all_question_embeddings.append(question_embeddings.detach().cpu())
            all_candidates_masks.append(candidates_mask.detach().cpu())
            all_node_maps.extend(node_maps)
            all_labels.extend(labels)

            # Only append similarity scores if they are not None
            if similarity_score is not None:
                all_similarity_scores.append(similarity_score.detach().cpu())  # Ensures tensor format
            else:
                print("Skipping batch with no similarity scores.")
                
    # Concatenate all batched data along the 0-axis (vertically)
    all_batched_subgraphs = torch.cat(all_batched_subgraphs, dim=0)
    all_question_embeddings = torch.cat(all_question_embeddings, dim=0)
    all_candidates_masks = torch.cat(all_candidates_masks, dim=0)
    all_similarity_scores = torch.cat(all_similarity_scores, dim=0) if all_similarity_scores else None

    # Save processed data to files
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
        "question_embeddings": question_embeddings,
        "candidates_masks": candidates_mask,
        "similarity_scores": similarity_scores,  # Leave as tensor if not None
        "node_maps": node_map,
        "labels": labels
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


import torch

def find_high_similarity_paths(saved_file_path, dataset, threshold=0.8):
    # Check if the dataset has the required attribute
    if not hasattr(dataset, 'id_to_node'):
        raise AttributeError("The dataset does not have 'id_to_node'. Ensure 'from_paths_activate' is set to True when initializing KGQADataset.")
    
    # Load saved data
    batched_subgraphs, question_embeddings, candidates_masks, similarity_scores, node_maps, labels = load_all_metadata(saved_file_path)
    
    # Prepare a list to store high-similarity paths
    high_similarity_paths = []

    # Iterate through each batch and filter by per-node similarity scores
    for i, (node_map, similarity_score, question_embedding) in enumerate(zip(node_maps, similarity_scores, question_embeddings)):
        if similarity_score is not None:
            # Ensure similarity_score is a tensor and apply sigmoid and thresholding
            similarity_score = torch.tensor(similarity_score) if isinstance(similarity_score, float) else similarity_score
            high_score_mask = torch.sigmoid(similarity_score) > threshold
            for idx, score in enumerate(similarity_score):
                if high_score_mask[idx]:  # Check if the node's score meets the threshold
                    node_idx = node_map.get(idx)
                    if node_idx is not None and node_idx in dataset.id_to_node:
                        original_entity = dataset.id_to_node[node_idx]
                        
                        # Find paths from the original entity in the main graph
                        paths = dataset.find_paths(dataset.G, original_entity, n=2)
                        high_similarity_paths.append((original_entity, paths, score.item()))
                    else:
                        print(f"Warning: Node index {idx} not found in node_map or not in id_to_node. Skipping this node.")
        else:
            print(f"Skipping batch {i} as it has no valid similarity scores.")
    
    return high_similarity_paths


def save_high_similarity_paths(high_similarity_paths, questions, output_file_path):
    # Prepare data for saving
    data_to_save = []
    
    # Iterate over high-similarity paths, adding question and paths for each entity
    for i, (entity, paths, score) in enumerate(high_similarity_paths):
        entry = {
            "entity": entity,
            "similarity_score": score,
            "question": questions[i],
            "paths": [
                [{"node": node, "relation": relation} for node, relation in path] for path in paths
            ]
        }
        data_to_save.append(entry)
    
    # Save to JSON file
    with open(output_file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)