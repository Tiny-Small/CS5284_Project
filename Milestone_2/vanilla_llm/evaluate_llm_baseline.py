import sys
sys.path.append("../../")

from tqdm import tqdm
import codecs, json

from LLM_baseline import *
from evaluation import *

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Evaluate LLM baseline')
parser.add_argument('--llm_path', type=str, help='path to llm')
parser.add_argument('--sbert_path', type=str, help='path to sbert')
parser.add_argument('--data_name', type=str, help='which data to evaluate')
args = parser.parse_args()

# load dataset
with open(f'../../Datasets/crawled/{args.data_name}_eval_crawled.json', 'r') as file:
    data = json.load(file)

# top k sentences will be retrieved using BM25
k = 50

# load model
llm = baseline_LLM(args.llm_path)

predicted_file = codecs.open(f"results/{args.data_name}_llm_baseline_predicted.tsv", 'w', 'utf-8')
predicted_file.write('gold\tpredicted' + '\n')
contexts_file = codecs.open(f"results/{args.data_name}_llm_baseline_contexts.txt", 'w', 'utf-8')
references = []
candidates = []

# iterate each query
for i in tqdm(range(len(data))):
    documents = [d for d in data[str(i)]['documents'] if d]
    if documents:
        rag = vanilla_RAG(data[str(i)]['query'], documents)
        top_k_sentences = rag.get_top_k_sentences(k=50)
        question = rag.question
        answer = data[str(i)]['answer']
        context = "\n".join(top_k_sentences)
        predicted = llm.predict(question, context)
        # write predicted and contexts
        predicted = ' '.join(predicted.split()) # remove multiple white spaces in prediction
        predicted_file.write(answer + '\t' + predicted + '\n')
        contexts_file.write("||".join(top_k_sentences) + "\n===========================\n") # separators used to divide data
        references.append(answer)
        candidates.append(predicted)
        # check empty prediction
        if not predicted.strip():
            print("Empty prediction at", i)
    else:
        print("No documents retrieved for", i)
evaluator = EvaluateMetrics(references, candidates, args.sbert_path)
results = evaluator.evaluate()
# write results
with open(f'results/{args.data_name}_llm_baseline_evaluation.json', 'w', encoding ='utf8') as json_file:
    json.dump(results, json_file, ensure_ascii = True, indent = 4)