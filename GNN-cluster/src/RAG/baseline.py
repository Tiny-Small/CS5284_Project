# bm25
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# llm
import torch
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

# general
import json, random
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx

# evaluation
from evaluation import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
To implement and evaluate
(1) BM25 as retrieval and (2) BM25+LLM as RAG
"""

class vanilla_RAG:
    """
    for each query
    """
    def __init__(self, question, answers, G):
        self.question = question
        self.entity = self.extract_entity_from_question(question)
        self.answers = answers
        self.sentences, self.paths = self.get_sentences(self.entity, G)
        self.top_k, self.top_k_nodes, self.scores = self.get_top_k_sentences(self.sentences, self.paths, len(self.answers)*3) ###
        
    def extract_entity_from_question(self, question):
        """
        Extract the entity that is enclosed in square brackets from the question.
        Example: "What city is [Paris] the capital of?" -> "Paris"
        """
        # assumes one entity of interest in each questin
        start = question.find('[') + 1
        end = question.find(']')
        if start == 0 or end == -1:
            raise ValueError(f"No entity found in the question: {question}")
        return question[start:end]
    
    def get_sentences(self, entity, G):
        """
        Dng 2-hop QA
        Get all possible 2-hop paths and construct sentences
        """
        paths = self.find_paths(G, entity, 2) + self.find_paths(G, entity, 1)

        sentences = []
        final_paths = []

        for tuple_list in paths:
            # Extract the last entity (candidate) in the path
            candidate_entity = tuple_list[-1][0]

            if candidate_entity != entity: # Avoid looping back to the query_entity
                # Create the sentence for the path
                sentence = ' '.join(f"{tup[0]} {tup[1]}" if tup[1] else tup[0] for tup in tuple_list)
                sentences.append(sentence)
                final_paths.append(tuple_list)
                
        return sentences, final_paths

    def find_paths(self, G, u, n):
        """
        Finds paths in a graph G starting from node u with until reaching a maximum length of n edges.

        Parameters:
        G (Graph): The nx graph where entities and relations are defined.
        u (str): The starting node for the paths.
        n (int): The maximum depth or length of paths in terms of edges.

        Returns:
        List of paths, where each path is a list of tuples (node, relation) representing
        the nodes and relations along the path.
        """

        if n == 0:
            return [[(u, None)]]

        paths = [
            [(u, G[u][neighbor]['relation'])] + path
            for neighbor in G.neighbors(u)
            for path in self.find_paths(G, neighbor, n - 1)
            if u not in [node for node, _ in path] # Avoid cycles
        ]
        return paths
    
    def get_top_k_sentences(self, sentences, paths, k):
        """
        Use BM25
        """
        # Tokenize the sentences into words (for BM25 input)
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

        # Initialize BM25 and train it on the tokenized sentences
        bm25 = BM25Okapi(tokenized_sentences)

        # Define the query and tokenize it
        query = self.question
        tokenized_query = word_tokenize(query.lower())

        # Get the BM25 scores for each sentence based on the query
        scores = bm25.get_scores(tokenized_query)

        # Rank sentences by score
        ranked_sentences = sorted(zip(scores, sentences), reverse=True, key=lambda x: x[0])
        ranked_paths = sorted(zip(scores, paths), reverse=True, key=lambda x: x[0]) # to extract last entry as the node entity

        return [sentence for score, sentence in ranked_sentences[:k]], [path[-1][0] for score, path in ranked_paths[:k]], {path[-1][0]: score for score, path in ranked_paths}
    
class baseline_LLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16)
        self.template ="""
You are doing extractive question answering. Strictly use the following pieces of context to answer the question. Answer directly, without elaboration. Output in comma-separated form.
{context}
Question: {question}
"""
        self.prompt_template = PromptTemplate.from_template(self.template)

    def predict(self, question, context):
        formatted_prompt = self.prompt_template.format(context = context, question = question)

        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt},
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

        outputs = self.model.generate(tokenized_chat, max_new_tokens=100).cpu()
        tokenized_chat = tokenized_chat.cpu() ###
        del tokenized_chat ###

        return self.tokenizer.decode(outputs[0]).split("<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]

def generate_nx_graph(path):
    """
    Constructs a networkx directed graph that includes both the original and reverse relations.
    Each edge includes the concatenated relations between the same pairs of entities.
    """
    df = pd.read_csv(path, sep='|', header=None, names=['entity1', 'relation', 'entity2'])

    # Remove duplicates
    df_unique = df.drop_duplicates() # 133582 edges after dedup

    # Define reverse relations and construct reverse edges
    reverse_relations = {
    'directed_by': 'directed',
    'written_by': 'written',
    'starred_actors': 'starring',
    'has_tags': 'is_tagged_to',
    'has_genre': 'is_genre_of',
    'has_imdb_rating': 'is_imdb_rating_of',
    'has_imdb_votes': 'is_imdb_votes_of',
    'in_language': 'language_of',
    'release_year': 'is_released_year_of'
    }

    reverse_rows = []
    for index, row in df_unique.iterrows():
        reverse_relation = reverse_relations[row['relation']]
        reverse_row = {'entity1': row['entity2'], 'relation': reverse_relation, 'entity2': row['entity1']}
        reverse_rows.append(reverse_row)

    df_reverse = pd.DataFrame(reverse_rows) # 133582 edges
    df_combined = pd.concat([df_unique, df_reverse], ignore_index=True) # 267164 edges

    # This step consolidates multiple edges between the same pair of entities into a single edge.
    # It concatenates all relation values associated with each pair of entities.
    df_final = df_combined.groupby(['entity1', 'entity2'], as_index=False).agg({
        'relation': ' and '.join
    }) # 249349 edges

    # Replace underscores in relation names
    df_final['relation'] = df_final['relation'].str.replace('_', ' ')

    G = nx.from_pandas_edgelist(df_final, source='entity1', target='entity2', edge_attr='relation', create_using=nx.DiGraph())

    return G

def main(sbert_path, llm, path_to_kb, path_to_qa, start_idx, end_idx):
    """
    To implement and evaluate (equal graph weightage)
    (1) BM25 as retrieval and (2) BM25+LLM as RAG
    """
    G = generate_nx_graph(path_to_kb)
    
    df = pd.read_csv(path_to_qa, sep='\t', header=None, names=['question', 'answer'])
    df['answer'] = df['answer'].apply(lambda x: x.split("|"))
    df = df.iloc[start_idx:end_idx]
    print(f"Total of {len(df)} questions")

    node_predictions = []
    node_labels = []
    node_scores = []
    llm_predictions = []
    groundtruths = []
    
    for idx, (q, a) in tqdm(enumerate(zip(df['question'], df['answer']))):
        rag = vanilla_RAG(q, a, G)

        # (1) BM25 as retrieval
        nodes = list(set([p[-1][0] for p in rag.paths]))
        node_pred = np.zeros(len(nodes), dtype = int)
        node_label = np.zeros(len(nodes), dtype = int)
        node_score = np.zeros(len(nodes))
        for i, x in enumerate(nodes):
            if x in rag.answers:
                node_label[i] = 1
            if x in rag.top_k_nodes[:len(a)]: # only consider up to number of answers for each question (avoid unecessary FP for BM25)
                node_pred[i] = 1
            node_score[i] = rag.scores[x]
        node_predictions.append(node_pred)
        node_scores.append(node_score)
        node_labels.append(node_label)
        
        # (2) BM25+LLM as RAG
        top_k_sentences = rag.top_k
        question = rag.question
        context = "\n".join(top_k_sentences)
        predicted = llm.predict(question, context)
        llm_predictions.append(predicted.strip())
        groundtruths.append(', '.join(a))
        
        print("="*100)
        print(f"Question {idx}:", q)
        print("Context:")
        print(context)
        print("Predicted:")
        print(predicted)
        print("Answer:", a)

    print("Evaluation:")
    
    # evaluate (1)
    evaluator_1 = RetrievalEvaluateMetrics(node_labels, node_predictions, node_scores)
    results_1 = evaluator_1.evaluate()    
    print(results_1)
    with open('results/Retrieval_bm25.json', 'w', encoding ='utf8') as json_file:
        json.dump(results_1, json_file, ensure_ascii = True, indent = 4)
    
    # evaluate (2)
    evaluator_2 = RAGEvaluateMetrics(groundtruths, llm_predictions, sbert_path)
    results_2 = evaluator_2.evaluate()
    print(results_2)
    with open('results/RAG_bm25.json', 'w', encoding ='utf8') as json_file:
        json.dump(results_2, json_file, ensure_ascii = True, indent = 4)
        
if __name__ == "__main__":
    """
    nohup python baseline.py --sbert_path 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' --kg_pth '../../data/raw/kb.txt' --qa_pth '../../data/raw/2-hop/qa_train.txt' --start_idx 50000 --end_idx 50200 --llm_path '/scratch/users/nus/e1329380/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/' > bm25_eval.log &

    tail -f bm25_eval.log
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate baseline RAGs')
    parser.add_argument('--sbert_path', type=str, help='path to sbert')
    parser.add_argument('--kg_pth', type=str, help='path to kg')
    parser.add_argument('--qa_pth', type=str, help='path to qa')
    parser.add_argument('--start_idx', type=int, help='starting idx')
    parser.add_argument('--end_idx', type=int, help='end idx (not inclusive)')
    parser.add_argument('--llm_path', type=str, help='path to llm')
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    # ensure reproducibility
    torch.manual_seed(2024)
    random.seed(2024)
    np.random.seed(2024)
    
    # load model
    llm = baseline_LLM(args.llm_path)
    main(args.sbert_path, llm, args.kg_pth, args.qa_pth, args.start_idx, args.end_idx)