import torch
from torch.utils.data import DataLoader, Subset
# import torch.nn.functional as F
import torch.nn as nn
from src.utils.config import load_config, validate_config
from src.utils.train import train_one_epoch, save_checkpoint
from src.utils.evaluation import evaluate
from src.utils.logging import log_metrics, save_config
from src.datasets.kgqa_dataset import KGQADataset
from src.datasets.data_utils import collate_fn
from src.models.gcn_model import GCNModel
from src.models.gat_model import GATModel

from sentence_transformers import SentenceTransformer


def get_model(model_name, config, question_embedding_dim):
    if model_name == "GCNModel":
        return GCNModel(
            in_channels=config['in_channels'],
            question_embedding_dim=question_embedding_dim,
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_GCNCov=config['num_layers'],
            PROC_QN_EMBED_DIM=config['PROC_QN_EMBED_DIM'],
            PROC_X_DIM=config['PROC_X_DIM']
        )
    elif model_name == "GATModel":
        return GATModel(
            config['in_channels'],
            config['hidden_channels'],
            config['out_channels'],
            config['num_layers']
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_pos_weight(dataloader):
    pos_cum = 0
    neg_cum = 0
    for _, _, stacked_labels, _, _ in dataloader:
        pos = stacked_labels.sum().item()
        neg = stacked_labels.numel() - pos
        pos_cum += pos
        neg_cum += neg
    return neg_cum, pos_cum


# Run it in GNN folder
def main(config_path='../config/train_config.yaml'):
    # Load and validate configuration
    config = load_config(config_path)
    required_keys = [
        'model', 'train', 'node_embed', 'idxes',
        'train_qa_data', 'test_qa_data', 'num_hops'
    ]
    validate_config(config, required_keys)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_folder = 'hf_model'
    encoding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", cache_folder=custom_folder)
    encoding_model.to(device)

    # Initialize Train Dataset and DataLoader
    train_dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config['node_embed'],
        path_to_idxes=config['idxes'],
        path_to_qa=config['train_qa_data'],
        k=config['num_hops']
    )

    sub_train_dataset = Subset(train_dataset, list(range(config['train']['start_idx'],
                                                         config['train']['end_idx'])))

    train_loader = DataLoader(
        sub_train_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=collate_fn,
        shuffle=True
    )

    sub_val_dataset = Subset(train_dataset, list(range(config['val']['start_idx'],
                                                       config['val']['end_idx'])))
    val_loader = DataLoader(
        sub_val_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=collate_fn,
        shuffle=True
    )

    # Initialize Test Dataset and DataLoader
    test_dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config['node_embed'],
        path_to_idxes=config['idxes'],
        path_to_qa=config['test_qa_data'],
        k=config['num_hops']
    )
    # Subset for testing a smaller set of data
    test_subset = Subset(test_dataset, list(range(config['test']['start_idx'],
                                                  config['test']['end_idx'])))
    test_loader = DataLoader(
        test_subset,
        batch_size=config['train']['batch_size'],
        collate_fn=collate_fn,
        shuffle=True
    )

    # Load job_name, equal_subgraph_weighting, and hits_at_k from config
    job_name = config['job_name']
    equal_subgraph_weighting = config['train']['equal_subgraph_weighting']
    hits_at_k = config['train']['hits_at_k']

    # Initialize model, optimizer, loss function, and variables for early stopping
    model = get_model(model_name=config['model']['name'],
                      config=config['model'],
                      question_embedding_dim=train_dataset.q_embeddings.size(-1)
                      ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    neg_cum, pos_cum = get_pos_weight(train_loader)
    reduction_type = 'none' if equal_subgraph_weighting else 'mean'
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_cum/pos_cum], device=device), reduction=reduction_type)
    # loss_fn = F.nll_loss  # Define your loss function here (use focal_loss if desired)

    # Wrap the model for data parallelism if more than one GPU is available (outside loop for efficiency)
    if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    best_val_f1 = 0  # For maximizing, start with the lowest possible value
    patience_counter = 0
    patience = config['train'].get('patience', 5)

    save_config(config=config, dir=f'saved_config', name=f'{job_name}_config.json')

    # Training and Evaluation Loop
    for epoch in range(config['train']['num_epochs']):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, equal_subgraph_weighting)
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} - Train Loss: {epoch_loss:.4f}")

        # Evaluate on training data (or use a separate validation set if available)
        # train_accuracy, train_precision, train_recall, train_f1, train_hits_at_k = evaluate(train_loader, model, device, equal_subgraph_weighting, hits_at_k)
        val_accuracy, val_precision, val_recall, val_f1, val_hits_at_k = evaluate(val_loader, model, device, equal_subgraph_weighting, hits_at_k)
        # test_accuracy, test_precision, test_recall, test_f1, test_hits_at_k = evaluate(test_loader, model, device, equal_subgraph_weighting, hits_at_k)

        # Log metrics
        # log_metrics(epoch, train_accuracy, train_precision, train_recall, train_f1, train_hits_at_k, log_dir = f'epoch_log/{job_name}', log_file='epoch_log/train_log.txt')
        log_metrics(epoch, val_accuracy, val_precision, val_recall, val_f1, val_hits_at_k, log_dir = f'epoch_log/{job_name}', log_file='validation_log.txt')
        # log_metrics(epoch, test_accuracy, test_precision, test_recall, test_f1, test_hits_at_k, log_dir = f'epoch_log/{job_name}', log_file='epoch_log/test_log.txt')

        # Print validation and test results
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}, Validation P/R/F1: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}')

        # Check for improvement in validation F1 for early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0  # Reset counter if improvement
            # Save best model
            save_checkpoint(model, optimizer, epoch+1, config, save_dir=f'checkpoints/{job_name}', best=True)
            print(f"Validation F1 improved to {val_f1:.4f}. Model checkpoint saved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation F1 for {patience_counter} epochs.")

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience_counter} epochs of no improvement.")
            break

        # Save checkpoint periodically
        save_checkpoint(model, optimizer, epoch+1, config, save_dir=f'checkpoints/{job_name}')

    # Evaluate and save metrics of test set
    test_accuracy, test_precision, test_recall, test_f1, test_hits_at_k = evaluate(test_loader, model, device, equal_subgraph_weighting, hits_at_k)
    log_metrics(epoch, test_accuracy, test_precision, test_recall, test_f1, test_hits_at_k, log_dir = f'epoch_log/{job_name}', log_file='test_log.txt')
    print(f'Epoch {epoch+1}, Test Accuracy: {test_accuracy:.4f}, Test P/R/F1: {test_precision:.3f}/{test_recall:.3f}/{test_f1:.3f}')

    # Save the final model
    # torch.save(model.state_dict(), 'final_model.pth')
    # print("Training completed and final model saved.")

if __name__ == '__main__':
    main()
