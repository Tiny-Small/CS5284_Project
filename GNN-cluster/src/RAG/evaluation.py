import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('punkt_tab')

"""
# how to use
# references: list of groundtruths
# candidates: list of predictions by LLM

from evaluation import *
import json

evaluator = RAGEvaluateMetrics(references, candidates, args.sbert_path)
results = evaluator.evaluate()
# write results
# result_path: path to store evaluation results
with open(result_path, 'w', encoding ='utf8') as json_file:
    json.dump(results, json_file, ensure_ascii = True, indent = 4)
"""

class RAGEvaluateMetrics:
    def __init__(self, references, candidates, sbert_path):
        self.references = references
        self.candidates = candidates
        self.sbert_model = SentenceTransformer(sbert_path)

    def compute_bleu(self):
        """
        Compute BLEU precision for each sentence using sentence-level BLEU score.
        """
        precisions = []
        smooth = SmoothingFunction().method1

        for ref, cand in zip(self.references, self.candidates):
            reference_tokens = [nltk.word_tokenize(ref.lower())]
            candidate_tokens = nltk.word_tokenize(cand.lower())

            # BLEU Precision (BLEU uses n-gram overlap precision by default)
            precision = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth)
            precisions.append(precision)

        return sum(precisions) / len(precisions)

    def compute_rouge_l(self):
        """
        Compute ROUGE-L precision, recall, and F1.
        """
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        precisions, recalls, f1_scores = [], [], []

        for ref, cand in zip(self.references, self.candidates):
            scores = rouge.score(ref.lower(), cand.lower())
            rouge_l = scores['rougeL']

            precisions.append(rouge_l.precision)
            recalls.append(rouge_l.recall)
            f1_scores.append(rouge_l.fmeasure)

        return sum(precisions) / len(precisions), sum(recalls) / len(recalls), sum(f1_scores) / len(f1_scores)

    def compute_bertscore(self, lang="en"):
        """
        Compute BERTScore (Precision, Recall, and F1).
        """
        P, R, F1 = bertscore(self.candidates, self.references, lang=lang, verbose=False) ###
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def compute_sbert(self):
        """
        Compute SBERT cosine similarity between reference and candidate sentences.
        """
        # Encode reference and candidate sentences using SBERT
        ref_embeddings = self.sbert_model.encode(self.references, convert_to_tensor=True)
        cand_embeddings = self.sbert_model.encode(self.candidates, convert_to_tensor=True)

        # Compute cosine similarity between each pair of reference and candidate
        cosine_similarities = util.pytorch_cos_sim(ref_embeddings, cand_embeddings)

        # Take the diagonal (similarity between ref[i] and cand[i])
        diagonal_similarities = cosine_similarities.diagonal()

        # Average similarity score
        average_similarity = diagonal_similarities.mean().item()

        return average_similarity

    def evaluate(self):
        """
        Run all evaluation metrics (BLEU, ROUGE-L, BERTScore) and return their results.
        :return: Dictionary containing BLEU, ROUGE-L, and BERTScore results.
        """
        results = {}

        # Compute BLEU
        bleu_p = self.compute_bleu()
        results['BLEU'] = {
            'Precision': bleu_p
        }

        # Compute ROUGE-L
        rouge_p, rouge_r, rouge_f1 = self.compute_rouge_l()
        results['ROUGE-L'] = {
            'Precision': rouge_p,
            'Recall': rouge_r,
            'F1': rouge_f1
        }

        # Compute BERTScore
        bert_p, bert_r, bert_f1 = self.compute_bertscore()
        results['BERTScore'] = {
            'Precision': bert_p,
            'Recall': bert_r,
            'F1': bert_f1
        }

        # Compute SBERT cosine similarity
        sbert_similarity = self.compute_sbert()
        results['SBERT'] = {
            'Cosine Similarity': sbert_similarity
        }

        return results

class RetrievalEvaluateMetrics:
    def __init__(self, node_labels, node_predictions, node_scores):
        self.references = node_labels
        self.candidates = node_predictions
        self.node_scores = node_scores

    def compute_p(self):
        total = 0
        for i in range(len(self.references)):
            total += precision_score(self.references[i], self.candidates[i], average='binary', zero_division=0)
        return total/len(self.references)

    def compute_r(self):
        total = 0
        for i in range(len(self.references)):
            total += recall_score(self.references[i], self.candidates[i], average='binary', zero_division=0)
        return total/len(self.references)

    def compute_f1(self):
        total = 0
        for i in range(len(self.references)):
            total += f1_score(self.references[i], self.candidates[i], average='binary', zero_division=0)
        return total/len(self.references)

    def compute_acc(self):
        total = 0
        for i in range(len(self.references)):
            total += sum(self.references[i] == self.candidates[i]) / len(self.references[i])
        return total/len(self.references)

    def compute_full_acc(self):
        total = 0
        for i in range(len(self.references)):
            total += 1 if sum(self.references[i] == self.candidates[i]) == len(self.references[i]) else 0
        return total/len(self.references)
    
    def compute_hit_k(self):
        total = 0
        for i in range(len(self.references)):

            sorted_indices = -self.node_scores[i].argsort() # descending order
            k = 3 ###
            top_k_indices = sorted_indices[:k]
            top_k_labels = self.references[i][top_k_indices]
        
            # Check if there's at least one positive label in the top-k predictions
            hits_at_k = int((top_k_labels == 1).any())

            total += 1 if hits_at_k else 0
            
        return total/len(self.references)
    
    def evaluate(self):
        """
        Run P/R/F1/hit@k/accuracy/full accuracy
        """
        results = {}

        # P/R/F1
        p = self.compute_p()
        results['precision'] = p
        r = self.compute_r()
        results['recall'] = r
        f1 = self.compute_f1()
        results['f1'] = f1

        # hit@k
        hit_k = self.compute_hit_k()
        results['hit@k'] = hit_k

        # accuracy/full accuracy
        acc = self.compute_acc()
        results['accuracy'] = acc
        full_acc = self.compute_full_acc()
        results['full accuracy'] = full_acc

        return results