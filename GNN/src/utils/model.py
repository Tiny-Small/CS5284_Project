from src.models.gcn_model import GCNModel
from src.models.gat_model import GATModel


def get_model(
    model_name,
    config,
):
    if model_name == "GCNModel":
        return GCNModel(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_layers=config["num_layers"],
            question_embedding_dim=config["question_embedding_dim"],
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
