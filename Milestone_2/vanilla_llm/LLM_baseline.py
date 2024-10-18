from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class vanilla_RAG:
    def __init__(self, question, documents):
        self.question = question
        self.sentences = self.get_sentences(documents)
    def get_sentences(self, documents):
        sentences = []
        for doc in documents:
            sentences.extend(sent_tokenize(doc))
        return sentences
    def get_top_k_sentences(self, k=100):
        """
        Use BM25
        """
        # Tokenize the sentences into words (for BM25 input)
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in self.sentences]

        # Initialize BM25 and train it on the tokenized sentences
        bm25 = BM25Okapi(tokenized_sentences)

        # Define the query and tokenize it
        query = self.question
        tokenized_query = word_tokenize(query.lower())

        # Get the BM25 scores for each sentence based on the query
        scores = bm25.get_scores(tokenized_query)

        # Rank sentences by score
        ranked_sentences = sorted(zip(scores, self.sentences), reverse=True, key=lambda x: x[0])

        return [sentence for score, sentence in ranked_sentences[:k]]
    
class baseline_LLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16)
        self.template ="""
Use the following pieces of context to answer the question. Answer directly, without elaboration. If no context provided, answer like a AI assistant.
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