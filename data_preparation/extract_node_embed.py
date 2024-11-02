import pandas as pd
import networkx as nx

from sentence_transformers import SentenceTransformer, util

"""
# testing
path_to_qa = '../Datasets/MetaQA_dataset/vanilla 2-hop/qa_train.txt'
qa_df = pd.read_csv(path_to_qa, sep='\t', header=None, names=['question', 'answer'])

G = generate_nx_graph('../Datasets/MetaQA_dataset/')
q_embeddings = model.encode([q.replace("[", "").replace("]", "") for q in qa_df['question']], batch_size=128, convert_to_tensor=True) # 32
qa_df['query_entity'] = qa_df['question'].apply(extract_entity_from_question) # similar to the one in functions.py
query_entities = qa_df['query_entity'].tolist()

for entity, embedding in zip(query_entities[:5000], q_embeddings[:5000]):
    result = find_best_embedding(G, entity, embedding)
    print(f"{i} Length of result for '{entity}': {len(result)}")
"""

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

def generate_nx_graph(path):
    """
    Constructs a networkx directed graph that includes both the original and reverse relations. 
    Each edge includes the concatenated relations between the same pairs of entities.
    """
    df = pd.read_csv(path+'kb.txt', sep='|', header=None, names=['entity1', 'relation', 'entity2'])
    
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
    # Number of entities: 43234
    # Number of edges: 249349
    # Number of distinct relations: 38
    # Distinct relations: {'release year', 'directed by and written by and starred actors', 'directed by and starred actors', 'has imdb rating', 'is genre of', 'has tags and is tagged to', 'directed by and written by', 'starred actors and starring', 'directed', 'directed and written', 'has imdb votes', 'written by and directed by', 'written and directed', 'in language and language of', 'language of', 'has genre', 'is tagged to', 'has imdb rating and has tags', 'directed and written and starring', 'starred actors', 'starring', 'has tags', 'directed and starring', 'written by and directed by and starred actors', 'written by and written', 'written by', 'in language', 'release year and has tags', 'written and directed and starring', 'written and starring', 'is released year of and is tagged to', 'directed by', 'is imdb rating of', 'is imdb rating of and is tagged to', 'written', 'is released year of', 'written by and starred actors', 'is imdb votes of'}

    return G

def find_paths(G, u, n):
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
        for path in find_paths(G, neighbor, n - 1)
        if u not in [node for node, _ in path] # Avoid cycles
    ]
    return paths

def find_best_embedding(G, query_entity, q_embedding):
    """
    Finds the best path embedding for each unique candidate based on cosine similarity.
    
    Parameters:
    G (Graph): The nx graph where entities and relations are defined.
    query_entity (str): The entity for which paths are being found.
    q_embedding (torch.Tensor): The embedding of the query entity.

    Returns:
    dict: A dictionary where keys are candidates and values are the best path embeddings.
    """

    paths = find_paths(G, query_entity, 2) + find_paths(G, query_entity, 1)

    sentences = []
    candidates = []

    for tuple_list in paths:
        # Extract the last entity (candidate) in the path
        candidate_entity = tuple_list[-1][0]
        
        if candidate_entity != query_entity: # Avoid looping back to the query_entity
            candidates.append(candidate_entity)
            # Create the sentence for the path
            sentence = ' '.join(f"{tup[0]} {tup[1]}" if tup[1] else tup[0] for tup in tuple_list)
            sentences.append(sentence)

    # Calculate path embeddings
    path_embeddings = model.encode(sentences, batch_size=128, convert_to_tensor=True)
    # Calculate cosine similarities
    cosine_scores = util.cos_sim(q_embedding, path_embeddings)[0]

    # Dictionary to store the best path embedding for each candidate
    best_embeddings = {}
    # Dictionary to store the highest cosine score for each candidate
    best_scores = {}

    for idx, candidate in enumerate(candidates):
        cosine_score = cosine_scores[idx].item()
        
        if candidate not in best_embeddings or cosine_score > best_scores[candidate]:
            best_scores[candidate] = cosine_score
            best_embeddings[candidate] = path_embeddings[idx]

    return best_embeddings