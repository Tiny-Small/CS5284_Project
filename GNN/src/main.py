import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import logging
from src.utils.config import load_config, validate_config
from src.utils.train import train, save_checkpoint
from src.utils.evaluation import evaluate
from src.utils.logging import log_metrics
from src.datasets.kgqa_dataset import KGQADataset
from src.datasets.data_utils import collate_fn
from src.models.gcn_model import GCNModel
from src.models.gat_model import GATModel
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_model(model_name, config):
    if model_name == "GCNModel":
        return GCNModel(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_layers=config["num_layers"],
            question_embedding_dim=384,  # Ensure this matches the actual output dimension
        )

    elif model_name == "GATModel":
        return GATModel(
            config["in_channels"],
            config["hidden_channels"],
            config["out_channels"],
            config["num_layers"],
            config["num_heads"],
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main(config_path="/hpctmp/e0315913/CS5284_Project/GNN/config/train_config.yaml"):
    # Load and validate configuration
    config = load_config(config_path)
    required_keys = [
        "model",
        "train",
        "node_embed",
        "idxes",
        "train_qa_data",
        "test_qa_data",
        "num_hops",
        "sentence_transformer_path",
    ]
    validate_config(config, required_keys)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SentenceTransformer model
    encoding_model = SentenceTransformer(config["sentence_transformer_path"])

    # Initialize Train Dataset and DataLoader
    logging.info("Initializing training dataset and dataloader.")
    train_dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config["node_embed"],
        path_to_idxes=config["idxes"],
        path_to_qa=config["train_qa_data"],
        path_to_ans_types="/hpctmp/e0315913/CS5284_Project/Datasets/MetaQA_dataset/vanilla 3-hop/qa_train_qtype.txt",
        train=True,
        k=config["num_hops"],
    )
    sub_train_data = torch.utils.data.Subset(train_dataset, list(range(1000)))

    train_loader = DataLoader(
        sub_train_data,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
    )

    # Initialize Test Dataset and DataLoader
    logging.info("Initializing test dataset and dataloader.")
    test_dataset = KGQADataset(
        encoding_model,
        path_to_node_embed=config["node_embed"],
        path_to_idxes=config["idxes"],
        path_to_qa=config["test_qa_data"],
        path_to_ans_types="/hpctmp/e0315913/CS5284_Project/Datasets/MetaQA_dataset/vanilla 3-hop/qa_test_qtype.txt",
        train=False,
        k=config["num_hops"],
    )
    test_subset = Subset(test_dataset, list(range(400)))
    test_loader = DataLoader(
        test_subset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
    )

    # Define Evaluation DataLoader with Smaller Batch Size
    eval_batch_size = (
        config["eval"]["batch_size"]
        if "eval" in config and "batch_size" in config["eval"]
        else 16
    )

    # Smaller batch-size DataLoader for evaluation
    train_eval_loader = DataLoader(
        sub_train_data,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # No need to shuffle for evaluation
    )

    test_eval_loader = DataLoader(
        test_subset, batch_size=eval_batch_size, collate_fn=collate_fn, shuffle=False
    )

    # Initialize model, optimizer, and loss function
    logging.info("Initializing model, optimizer, and loss function.")
    model = get_model(config["model"]["name"], config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["learning_rate"]
    )

    # Training and Evaluation Loop
    for epoch in range(2):
        logging.info(f"Starting epoch {epoch+1}/{config['train']['num_epochs']}")

        # Train for one epoch
        epoch_loss = train(model, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}")

        # Evaluate on training and test sets with smaller batches
        logging.info("Evaluating on training data.")
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(
            train_eval_loader, model, device
        )
        logging.info("Evaluating on test data.")
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(
            test_eval_loader, model, device
        )

        # Log metrics
        log_metrics(
            epoch,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            log_file=config.get("validation_log", "validation_log.txt"),
        )
        log_metrics(
            epoch,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            log_file=config.get("test_log", "test_log.txt"),
        )

        # Print validation and test results
        logging.info(
            f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation P/R/F1: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}"
        )
        logging.info(
            f"Test Accuracy: {test_accuracy:.4f}, "
            f"Test P/R/F1: {test_precision:.3f}/{test_recall:.3f}/{test_f1:.3f}"
        )

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            config,
            save_dir=config.get("save_dir", "checkpoints"),
        )
        logging.info(f"Checkpoint saved for epoch {epoch+1}")

    # Save the final model
    final_model_path = config.get("final_model_path", "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training completed. Final model saved at {final_model_path}")


if __name__ == "__main__":
    main()
