import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.config import load_config, validate_config
from src.utils.train import train_one_epoch, save_checkpoint
from src.utils.evaluation import evaluate
from src.utils.logging import log_metrics, save_config
from src.datasets.kgqa_dataset import KGQADataset
from src.datasets.data_utils import collate_fn
from src.models.gcn_model import GCNModel
from src.models.gat_model import GATModel
from src.models.rgcn_model import RGCNModel
from src.models.threshold_model import ThresholdedModel
# from utils.config import load_config, validate_config
# from utils.train import train_one_epoch, save_checkpoint
# from utils.evaluation import evaluate
# from utils.logging import log_metrics, save_config
# from datasets.kgqa_dataset import KGQADataset
# from datasets.data_utils import collate_fn
# from models.gcn_model import GCNModel
# from models.gat_model import GATModel
# from models.rgcn_model import RGCNModel
# from models.threshold_model import ThresholdedModel

from sentence_transformers import SentenceTransformer


def get_model(model_name, config, question_embedding_dim, num_relations):
    if model_name == "GCNModel":
        return GCNModel(
            in_channels=config['in_channels'],
            question_embedding_dim=question_embedding_dim,
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_GCNCov=config['num_layers'],
            PROC_QN_EMBED_DIM=config['PROC_QN_EMBED_DIM'],
            PROC_X_DIM=config['PROC_X_DIM'],
            output_embedding=config['output_embedding']
        )
    elif model_name == "RGCNModel":
        return RGCNModel(
            node_dim=config['in_channels'],
            question_dim=question_embedding_dim,
            hidden_dim=config['hidden_channels'],
            num_relations=num_relations,
            output_dim=config['out_channels'],
            num_rgcn=config['num_layers'],
            reduced_qn_dim=config['reduced_qn_dim'],
            reduced_node_dim=config['reduced_node_dim'],
            output_embedding=config['output_embedding']
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
    num_relations = train_dataset.num_relations # extract the num_relation from the entire graph
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

    # Load job_name, equal_subgraph_weighting, threshold_value, and hits_at_k from config
    job_name = config['job_name']
    equal_subgraph_weighting = config['train']['equal_subgraph_weighting']
    hits_at_k = config['train']['hits_at_k']
    threshold_model_activate = config['threshold_model_activate']
    threshold_value = config['threshold_value']

    # Initialize model, optimizer, scheduler, loss function, and variables for early stopping
    model = get_model(model_name=config['model']['name'],
                      config=config['model'],
                      question_embedding_dim=train_dataset.q_embeddings.size(-1),
                      num_relations=num_relations
                      ).to(device)

    if threshold_model_activate:
        model = ThresholdedModel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, equal_subgraph_weighting)
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} - Train Loss: {epoch_loss:.4f}")

        # Evaluate on training data (or use a separate validation set if available)
        train_metrics = evaluate(train_loader, model, device, equal_subgraph_weighting, threshold_value, hits_at_k)
        val_metrics = evaluate(val_loader, model, device, equal_subgraph_weighting, threshold_value, hits_at_k)

        scheduler.step(val_metrics.precision)

        # Log metrics
        log_metrics(epoch, train_metrics.accuracy, train_metrics.precision, train_metrics.recall, train_metrics.f1,
                    train_metrics.hits_at_k, train_metrics.full_accuracy,
                    log_dir = f'epoch_log/{job_name}', log_file='train_log.txt')
        log_metrics(epoch, val_metrics.accuracy, val_metrics.precision, val_metrics.recall, val_metrics.f1,
                    val_metrics.hits_at_k, val_metrics.full_accuracy,
                    log_dir = f'epoch_log/{job_name}', log_file='validation_log.txt')

        # Print train and validation results
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy/Full: {train_metrics.accuracy:.4f}/{train_metrics.full_accuracy:.4f}, Train P/R/F1: {train_metrics.precision:.3f}/{train_metrics.recall:.3f}/{train_metrics.f1:.3f}')
        print(f'Validation Accuracy/Full: {val_metrics.accuracy:.4f}/{val_metrics.full_accuracy:.4f}, Validation P/R/F1: {val_metrics.precision:.3f}/{val_metrics.recall:.3f}/{val_metrics.f1:.3f}')

        # Check for improvement in validation F1 for early stopping
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            patience_counter = 0  # Reset counter if improvement
            # Save best model
            save_checkpoint(model, optimizer, epoch+1, config, save_dir=f'checkpoints/{job_name}', best=True)
            print(f"Validation F1 improved to {val_metrics.f1:.4f}. Model checkpoint saved.")
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
    test_metrics = evaluate(test_loader, model, device, equal_subgraph_weighting, hits_at_k)
    log_metrics(epoch, test_metrics.accuracy, test_metrics.precision, test_metrics.recall, test_metrics.f1,
                test_metrics.hits_at_k, test_metrics.full_accuracy,
                log_dir = f'epoch_log/{job_name}', log_file='test_log.txt')
    print(f'Epoch {epoch+1}, Test Accuracy/Full: {test_metrics.accuracy:.4f}, {test_metrics.full_accuracy:.4f}, Test P/R/F1: {test_metrics.precision:.3f}/{test_metrics.recall:.3f}/{test_metrics.f1:.3f}')

if __name__ == '__main__':
    main()
