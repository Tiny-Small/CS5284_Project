from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse, json, newspaper

parser = argparse.ArgumentParser(description='Crawl articles')
parser.add_argument('--data_name', type=str, help='which data to crawl')
args = parser.parse_args()

config = newspaper.configuration.Configuration(fetch_images = False) # smt wrong with source code, edited directly at source code, included fetch_images as input parameter 
config.update(request_timeout = 3)

# load dataset
if args.data_name == "frames":
    ds = load_dataset("google/frames-benchmark")
    data = pd.DataFrame.from_dict(ds["test"])
elif args.data_name == "ms_marco":
    ds = load_dataset("microsoft/ms_marco", "v2.1") # v1.1
    # first 500 row only
    data = pd.DataFrame.from_dict(ds["validation"])

out = {}
for i in tqdm(range(len(data))):
    
    if args.data_name == "frames":
        links = eval(data['wiki_links'][i])
        query = data['Prompt'][i]
        answer = data['Answer'][i]
        query_type = data['reasoning_types'][i]
    elif args.data_name == "ms_marco":
        # early break up to 500 rows only
        if i>=500:
            break
        links = data["passages"][i]["url"]
        query = data["query"][i]
        answer = data["answers"][i][0]
        query_type = data['query_type'][i]

    # crawl articles
    documents = []
    for link in links:
        try:
            article = newspaper.article(link)#, config = config)
            documents.append(article.text)
        except:
            print(f"Failed to retrieve {i}th article.")
            documents.append(None)
        out[i] = {"query": query, "answer": answer, "documents": documents, "query_type": query_type}

# write crawled
with open(f'crawled/{args.data_name}_eval_crawled.json', 'w', encoding ='utf8') as json_file:
    json.dump(out, json_file, ensure_ascii = True, indent = 4)