import os
import torch
import logging
from src.utils.config import load_config, validate_config
from src.utils.model import get_model
from src.utils.train import train, save_checkpoint
from src.utils.evaluation import evaluate
from src.utils.logging import log_metrics
from src.datasets.data_utils import data_loader
from sentence_transformers import SentenceTransformer

os.chdir("/hpctmp/e0315913/CS5284_Project/GNN")


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(config_path="/hpctmp/e0315913/CS5284_Project/GNN/config/train_config.yaml"):
    # Load and validate configuration
    config = load_config(config_path)
    required_keys = [
        "model",
        "train",
        "val",
        "test",
        "node_embed",
        "idxes",
        "train_qa_data",
        "test_qa_data",
        "train_ans_types",
        "num_hops",
        "save_dir",
        "sentence_transformer_path",
    ]
    validate_config(config, required_keys)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SentenceTransformer model
    encoding_model = SentenceTransformer(config["sentence_transformer_path"])

    # Initialize Train Dataset and DataLoader
    logging.info("Initializing dataset and dataloaders.")
    train_loader = data_loader("train", encoding_model, config)
    val_loader = data_loader("val", encoding_model, config)
    test_loader = data_loader("test", encoding_model, config)

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
        logging.info("Evaluating on val data.")
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(
            val_loader, model, device
        )
        logging.info("Evaluating on test data.")
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(
            test_loader, model, device
        )

        # Log metrics
        log_metrics(
            epoch,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            log_file=config.get("val_log_dir", "logs/validation_log.txt"),
        )
        log_metrics(
            epoch,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            log_file=config.get("test_log_dir", "logs/test_log.txt"),
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
