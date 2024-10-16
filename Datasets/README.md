HuggingFace data will be stored here.

### KGQA datasets used for training GNN

### Crawl Articles from google/frames and microsoft/ms_marco for evaluation
As huggingface does not provide the text of the articles used for each query, we had to crawl for the articles' text.

**Run the following code:**
```
python crawl_articles.py --data_name 'ms_marco' 2>&1 | tee ms_marco_crawl.log
```
or
```
python crawl_articles.py --data_name 'frames' 2>&1 | tee frames_crawl.log
```