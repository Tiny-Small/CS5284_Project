# CS5284 Project: Knowledge Graph Question Answering Using RGCN and LLM

## Abstract
This project explores Knowledge Graph Question Answering (KGQA) using a Retrieval-Augmented Generation (RAG) framework integrating a Relational Graph Convolutional Network (RGCN) and a Large Language Model (LLM). The RGCN retrieves answer candidates through multi-hop reasoning over the knowledge graph (KG), while the LLM generates accurate, context-aware answers. Using the MetaQA dataset, we evaluated embedding strategies, data distribution, and model configurations. Results show that RGCN outperforms traditional methods like BM25 for multi-hop queries and that using a consistent embedding model for nodes and questions enhances GNN performance. Additionally, explicit pathing may be unnecessary, as GNN message passing effectively captures pathing information.

## Overview
This project focuses on developing a Retrieval-Augmented Generation (RAG) system for Knowledge Graph Question Answering (KGQA). The final implementation integrates a Relational Graph Convolutional Network (RGCN) for candidate retrieval and a Large Language Model (LLM) for answer generation.

Extensive experiments on the MetaQA dataset were conducted to optimize the RAG system (GNN+LLM), including hyperparameter tuning and evaluation of node embedding strategies.

Initial explorations, such as building a knowledge graph manually and experimenting with GraphRAG, are documented in other folders.

## Directory Structure
This project is organized as follows:

- `Milestone_1/KG_REBEL/`: Initial exploration of manual knowledge graph construction.
- `Milestone_1/GraphRAG/`: Experiments using GraphRAG for retrieval tasks.
- `Milestone_2/`: Early evaluation of the RAG system.
- `data_exploration/`: Analysis of the MetaQA datasets.
- `data_preparation/`: Data preprocessing and embedding experiments.
- `Datasets/crawled/`: Crawled datasets for initial KG explorations.
- `Datasets/MetaQA_dataset/`: MetaQA datasets used in experiments.
- `GNN/`: Initial GNN pipeline development.
- `GNN-cluster/`: Finalized GNN pipeline with training scripts.
- `LLM-GNN/`: Integration of GNN for retrieval with LLM for answer generation.
- `training-graph-models/`: Graph model training and development

## Quickstart Guide
To explore the main implementation, navigate to the `GNN-Cluster` folder. Detailed setup instructions are provided there.

## Dependencies
Before starting, ensure you have:
- Python 3.10.6
- Required packages listed in the `requirements.txt` of respective folders.
- Installing `torch-scatter`
  -  Install torch-scatter separately by following the instructions at [torch-scatter GitHub](https://github.com/rusty1s/pytorch_scatter/issues/424)
- PyTorch Installation
  - Use the specific installation command for your system from [PyTorch.org](https://pytorch.org/get-started/locally/).

## Credits
This project was developed by:
- [tyeeying](https://github.com/tyeeying)
- [Jocely22](https://github.com/Jocely22)
- [Park Jinah](https://github.com/ParkJinah99)
- [Tiny-Small](https://github.com/Tiny-Small)
