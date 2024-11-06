import json, random, torch
import pandas as pd

from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import networkx as nx

from sentence_transformers import util, SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# custom_folder = '../hf_model' ### you might need to modify this
# model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", cache_folder=custom_folder)
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.to(device)

class KGQADataset(torch.utils.data.Dataset):
    def __init__(self, path_to_node_embed, path_to_idxes, path_to_qa, path_to_kb, from_paths_activate, k=3):
        """
        Initialize without precomputed subgraphs. Computes k-hop subgraphs on-the-fly.
        """
        self.from_paths_activate = from_paths_activate # needed in __getitem__
        
        if self.from_paths_activate:
            # Load the main graph data
            self.G = self.generate_nx_graph(path_to_kb)
            relations = [data['relation'] for _,_,data in self.G.edges(data=True)]
            relation_mapping = {relation: int(index) for index, relation in enumerate(set(relations))}
            self.pyg_graph = from_networkx(self.G)
            self.pyg_graph.edge_attr = torch.tensor([relation_mapping[r] for r in relations])
            
            # get node ids/names
            self.id_to_node = {}
            self.node_to_id = {}
            for i, n in enumerate(list(self.G.nodes)):
                self.id_to_node[i] = n
                self.node_to_id[n] = i

            # Store the global number of unique relations
            self.num_relations = len(set(relations))
            # print(f"Number of unique relations: {self.num_relations}")

        # OLD
        else:
            # Load the main graph data
            self.loaded_entity_to_idx, self.loaded_edge_index, self.loaded_relations = self.load_data_json(path_to_idxes)
            self.data = self.create_data_object(self.loaded_edge_index, self.loaded_relations, self.loaded_entity_to_idx)
    
            # Store the global number of unique relations
            self.num_relations = len(set(self.loaded_relations))
            self.k = k

            # Load node2vec embeddings
            self.node2vec_embeddings = self.load_node2vec_embeddings(path_to_node_embed)

        # Load question and answer data
        self.df = pd.read_csv(path_to_qa, sep='\t', header=None, names=['question', 'answer'])
        self.df['answer'] = self.df['answer'].apply(lambda x: x.split("|"))

        # Load sentence embeddings
        self.q_embeddings = model.encode(
            [q.replace("[", "").replace("]", "") for q in self.df['question']],
            batch_size=128,
            convert_to_tensor=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the question and answers from the DataFrame
        row = self.df.iloc[idx]
        question, answers = row['question'], row['answer']

        # Step 1: Extract the entity from the question (entity marked in square brackets)
        entity = self.extract_entity_from_question(question)
        # if entity not in self.loaded_entity_to_idx:
        #     raise ValueError(f"Entity {entity} not found in node index.")
        
        # Step 2: Get the question embedding
        question_embedding = self.q_embeddings[idx]

        if self.from_paths_activate:
            # Step 3: Find paths and get node embeddings
            result = self.find_best_embedding(self.G, entity, question_embedding)

            # Step 4: Construct subgraphs from paths found
            subset_nodes = [self.node_to_id[e] for e in list(result.keys())+[entity]]
            node_map = {} # old to new
            new_to_old = {}
            for new_idx, old_idx in enumerate(subset_nodes):
                node_map[old_idx] = new_idx
                new_to_old[new_idx] = old_idx
            query_edge_index, query_edge_attr = subgraph(subset=subset_nodes, edge_index=self.pyg_graph.edge_index, edge_attr=self.pyg_graph.edge_attr, relabel_nodes=True)
            subgraph_data = Data(edge_index=query_edge_index, edge_attr=query_edge_attr)
            subgraph_data.num_nodes = len(subset_nodes)

            # Step 5: Get node embeddings for subgraphs
            node_embeddings = []
            for i in range(subgraph_data.num_nodes):
                ent = self.id_to_node[new_to_old[i]]
                if ent == entity:
                    node_embeddings.append(question_embedding)
                else:
                    node_embeddings.append(result[ent])
            subgraph_data.x = torch.stack(node_embeddings)

            # Step 6: Get labels
            labels = torch.zeros(len(node_map), dtype=torch.long)
            for ans in answers:
                if ans in result.keys():
                    ans_idx = self.node_to_id[ans]
                    if ans_idx in node_map:
                        labels[node_map[ans_idx]] = 1
            assert sum(labels) == len(answers), f"Different number of labels (sum(labels)) and answers (len(answers))"

        # OLD
        else:
            entity_node = self.loaded_entity_to_idx[entity]
    
            # Step 3: Compute the k-hop subgraph around the entity dynamically
            subset_node_indices, sub_edge_index, _, edge_mask = self.get_k_hop_subgraph(entity_node)
    
            # Step 4: Construct the subgraph based on these subset indices
            subgraph_data, node_map = self.construct_subgraph(subset_node_indices, sub_edge_index, edge_mask)
    
            # Step 5: Get the labels
            labels = self.get_labels(answers, node_map)
    
            # Step 6: Add node2vec embeddings to the subgraph data
            subgraph_data.x = self.get_node_embeddings(node_map)

        return subgraph_data, question_embedding, labels, node_map

    def get_k_hop_subgraph(self, entity_node):
        """
        Compute the k-hop subgraph dynamically for a given entity node.
        """
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=entity_node,
            num_hops=self.k,
            edge_index=self.data.edge_index,
            relabel_nodes=True
        )
        return subset, sub_edge_index, mapping, edge_mask

    def construct_subgraph(self, subset_node_indices, sub_edge_index, edge_mask):
        """
        Construct a subgraph Data object for the given subset of nodes and edges.
        """
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset_node_indices)}

        # # Create subgraph data object
        sub_edge_attr = self.data.edge_attr[edge_mask]
        subgraph_data = Data(edge_index=sub_edge_index, edge_attr=sub_edge_attr) # sub_edge_index is get_k_hop_subgraph's sub_edge_index
        return subgraph_data, node_map

    def load_data_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['entity_to_idx'], data['edge_index'], data['relations']

    def load_data_pt(self, filename):
        """
        Load the list of subsets (k-hop subgraph node indices) from the .pt file.
        """
        data = torch.load(filename)
        if isinstance(data, list):
            return data  # Return the list of subsets
        else:
            raise ValueError("Expected a list of k-hop subgraph node indices.")

    def create_data_object(self, edge_index, relations, entity_to_idx):
        unique_relations = list(set(relations))
        relation_mapping = {relation: index for index, relation in enumerate(unique_relations)}

        edge_index = torch.tensor(edge_index).t().contiguous()
        # Make the graph undirected by adding reverse edges
        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # comment out if want directed
        edge_attr = torch.tensor([relation_mapping[rel] for rel in relations])
        # Since we now have more edges (two for each undirected edge), we concat them
        undirected_edge_attr = torch.cat([edge_attr, edge_attr], dim=0) # comment out if want directed

        # Data(edge_index=undirected_edge_index, edge_attr=undirected_edge_attr, num_nodes=len(entity_to_idx))
        # Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(entity_to_idx))
        return Data(edge_index=undirected_edge_index, edge_attr=undirected_edge_attr, num_nodes=len(entity_to_idx))

    def load_node2vec_embeddings(self, file_path, embedding_dim=64):
        embeddings_dict = {}

        with open(file_path, 'r') as f:
            # Skip the first row
            next(f)

            for line in f:
                parts = line.strip().split()

                # The entity is everything before the embedding, so we use -embedding_dim
                entity = " ".join(parts[:-embedding_dim])  # Join all words before the embedding dimensions
                embedding = list(map(float, parts[-embedding_dim:]))  # Convert last parts to float

                # Store in the dictionary
                embeddings_dict[entity] = embedding

        return embeddings_dict

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

    # def get_k_hop_subgraph(self, node_idx):
    #     """
    #     Get the k-hop subgraph centered around the given node index.
    #     node_idx (int): Index of the node representing the entity.
    #     """
    #     # Extract k-hop subgraph from the full graph
    #     node_idx = torch.tensor([node_idx], dtype=torch.long)
    #     subset, sub_edge_index, _, _ = k_hop_subgraph(
    #         node_idx=node_idx,
    #         num_hops=self.k,
    #         edge_index=self.data.edge_index,
    #         relabel_nodes=True
    #     )

    #     # Create a subgraph Data object
    #     # subgraph = Data(x=self.data.x[subset], edge_index=sub_edge_index)
    #     subgraph = Data(edge_index=sub_edge_index)

    #     # Create a mapping from original node indices to subgraph indices
    #     node_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(subset)}

    #     return subgraph, node_map

    def get_labels(self, answers, node_map):
        labels = torch.zeros(len(node_map), dtype=torch.long)
        for ans in answers:
            if ans in self.loaded_entity_to_idx:
                ans_idx = self.loaded_entity_to_idx[ans]
                if ans_idx in node_map:
                    labels[node_map[ans_idx]] = 1
        return labels

    # def get_node_embeddings(self, node_map):
    #     embeddings = [[0.0] * len(self.node2vec_embeddings[next(iter(self.node2vec_embeddings))])]*len(node_map)
    #     idx_to_entity = {v: k for k, v in self.loaded_entity_to_idx.items()}

    #     for ori, new in node_map.items():
    #         if idx_to_entity[ori] in self.node2vec_embeddings:
    #             embeddings[new] = self.node2vec_embeddings[idx_to_entity[ori]]
    #     return torch.tensor(embeddings, dtype=torch.float)

    def get_node_embeddings(self, node_map, embedding_dim=64, random_init=False):
        """
        Get node embeddings from node2vec. If node not found or random_init is True, create random embeddings.
        """
        embeddings = [[0.0] * embedding_dim] * len(node_map)  # Initialize all embeddings with zeros
        idx_to_entity = {v: k for k, v in self.loaded_entity_to_idx.items()}

        for ori, new in node_map.items():
            if random_init:
                # Randomly initialize embeddings if random_init is True
                embeddings[new] = [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
            elif idx_to_entity[ori] in self.node2vec_embeddings:
                embeddings[new] = self.node2vec_embeddings[idx_to_entity[ori]]
            # else:
                # Create random embedding for missing nodes if node2vec embedding doesn't exist
                # embeddings[new] = [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]

        return torch.tensor(embeddings, dtype=torch.float)

    def generate_nx_graph(self, path):
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
        # Number of entities: 43234
        # Number of edges: 249349
        # Number of distinct relations: 38
        # Distinct relations: {'release year', 'directed by and written by and starred actors', 'directed by and starred actors', 'has imdb rating', 'is genre of', 'has tags and is tagged to', 'directed by and written by', 'starred actors and starring', 'directed', 'directed and written', 'has imdb votes', 'written by and directed by', 'written and directed', 'in language and language of', 'language of', 'has genre', 'is tagged to', 'has imdb rating and has tags', 'directed and written and starring', 'starred actors', 'starring', 'has tags', 'directed and starring', 'written by and directed by and starred actors', 'written by and written', 'written by', 'in language', 'release year and has tags', 'written and directed and starring', 'written and starring', 'is released year of and is tagged to', 'directed by', 'is imdb rating of', 'is imdb rating of and is tagged to', 'written', 'is released year of', 'written by and starred actors', 'is imdb votes of'}
    
        return G

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

    def find_best_embedding(self, G, query_entity, q_embedding):
        """
        Finds the best path embedding for each unique candidate based on cosine similarity.
        
        Parameters:
        G (Graph): The nx graph where entities and relations are defined.
        query_entity (str): The entity for which paths are being found.
        q_embedding (torch.Tensor): The embedding of the query entity.
    
        Returns:
        dict: A dictionary where keys are candidates and values are the best path embeddings.
        """
    
        paths = self.find_paths(G, query_entity, 2) + self.find_paths(G, query_entity, 1)
    
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