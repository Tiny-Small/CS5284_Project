# GNN-cluster
This folder contains the main implementation of the project, including all necessary configurations, data preparation scripts, model training code, and evaluations. Below is an overview of the directory structure:

## Directory Structure
- `config/`: Contains configuration files used for training and testing.
- `data/`: Stores preprocessed data and datasets used for model training and evaluation.
- `notebooks/`: Jupyter notebooks for data exploration and preparation.
- `src/`: Main source code for the implementation.
  - `checkpoints/`: Stores the final trained model checkpoints.
  - `epoch_log/`: Logs for tracking performance across epochs.
  - `LLM_GNN/`: Code for incorporating the LLM with the GNN for RAG evaluation.
  - `models/`: Contains the GNN models (e.g., RGCN) and auxiliary components like learnable margin modules.
  - `my_datasets/`: Files related to data loaders and dataset handling.
  - `RAG/`: Scripts for evaluating the RAG system combining GNN and LLM.
  - `saved_config/`: Configuration files tested during experimentation.
  - `utils/`: Utility scripts for configuration, evaluation, logging, and training.
  - `main.py`: Entry point to start model training and related tasks.


## How to Use
- Install the required packages listed in the `requirements.txt`.
- Configure the training parameters using the files in `config/`.
- Run the following to start training:
```bash
cd src
python main.py
```
