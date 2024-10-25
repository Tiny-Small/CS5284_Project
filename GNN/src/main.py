import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from src.utils.config import load_config, validate_config
from src.utils.train import train, save_checkpoint
from src.utils.evaluation import evaluate
from src.utils.logging import log_metrics
from src.datasets.kgqa_dataset import KGQADataset
from src.datasets.data_utils import collate_fn
from src.models.gcn_model import GCNModel
from src.models.gat_model import GATModel

from sentence_transformers import SentenceTransformer


def get_model(model_name, config):
    if model_name == "GCNModel":
        return GCNModel(
            config['in_channels'], 
            config['hidden_channels'], 
            config['out_channels'], 
            config['num_layers']
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

def main(config_path='/hpctmp/e0315913/CS5284_Project/GNN/config/train_config.yaml'):
    # Load and validate configuration
    config = load_config(config_path)
    required_keys = [
        'model', 'train', 'node_embed', 'idxes', 
        'train_qa_data', 'test_qa_data', 'num_hops'
    ]
    validate_config(config, required_keys)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoding_model = SentenceTransformer("/hpctmp/e0315913/CS5284_Project/GNN/src/models/models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1/snapshots/2d981ed0b0b8591b038d472b10c38b96016aab2e")

    # Initialize Train Dataset and DataLoader
    train_dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config['node_embed'], 
        path_to_idxes=config['idxes'], 
        path_to_qa=config['train_qa_data'], 
        k=config['num_hops']
    )
    train_loader = DataLoader(
        train_dataset, 
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
    test_subset = Subset(test_dataset, list(range(400)))
    test_loader = DataLoader(
        test_subset, 
        batch_size=config['train']['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=True
    )

    # Initialize model, optimizer, and loss function
    model = get_model(config['model']['name'], config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    loss_fn = F.nll_loss  # Define your loss function here (use focal_loss if desired)

    # Training and Evaluation Loop
    for epoch in range(config['train']['num_epochs']):
        # Train for one epoch
        epoch_loss = train(model, train_loader, optimizer, loss_fn, config, device)
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} - Train Loss: {epoch_loss:.4f}")

        # Evaluate on training data (or use a separate validation set if available)
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(train_loader, model, device)
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(test_loader, model, device)

        # Log metrics
        log_metrics(epoch, val_accuracy, val_precision, val_recall, val_f1, log_file='validation_log.txt')
        log_metrics(epoch, test_accuracy, test_precision, test_recall, test_f1, log_file='test_log.txt')

        # Print validation and test results
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}, Validation P/R/F1: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}')
        print(f'Test Accuracy: {test_accuracy:.4f}, Test P/R/F1: {test_precision:.3f}/{test_recall:.3f}/{test_f1:.3f}')

        # Save checkpoint periodically
        save_checkpoint(model, optimizer, epoch+1, config)

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training completed and final model saved.")

if __name__ == '__main__':
    main()
