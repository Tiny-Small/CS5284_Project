## RAG system
We investigated two kinds of RAG systems.

### BM25+LLM (traditional retrieval)
Start off we setting up,
```
pip install sentence-transformers langchain transformers bitsandbytes accelerate torch
pip install rank_bm25 bert-score rouge-score nltk scikit-learn
pip install networkx
```
Run in python,
``` python
nltk.download('punkt')
nltk.download('punkt_tab')
```
Next you will need to download "meta-llama/Llama-3.1-8B-Instruct" first, from huggingface to a directory of your choice.
``` python
# might need to pip install huggingface-hub first
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", local_dir = <YOUR DIRECTORY OF CHOICE>)
```
To implement and evaluate (1) BM25 as retrieval and (2) BM25+LLM as RAG:
```
python baseline.py --sbert_path 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' --kg_pth '../../data/raw/kb.txt' --qa_pth '../../data/raw/2-hop/qa_train.txt' --start_idx 50000 --end_idx 50200 --llm_path <YOUR DIRECTORY OF CHOICE>
```
The evaluation is done on the validation dataset subset we chosen. And results are stored in the `results` folder as `RAG_bm25.json` and `Retrieval_bm25.json`.

### GNN+LLM
The GNN used here is RGCN and we used SBERT-entity as node embedding.
