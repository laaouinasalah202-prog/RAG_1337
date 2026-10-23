from sentence_transformers import SentenceTransformer
import json
import tqdm
import numpy as np
from typing import Dict, List
import os
import pickle


class Semantic_search:
    def __init__(self, model_name):
        
        with open("data/processed/index.json", "r") as f:
            self.chunks_index = json.load(f)
        self.model_name = model_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.path_embeddings = "data/processed/embeddings.pkl"
        self.chunk_embedding = []
        
    def __read_chunk(self, chunk):
        with open(chunk["path"], "r", encoding="utf-8") as f:
            chunk_string = f.read()[chunk["first_character_index"]:chunk["last_character_index"]]
        return chunk_string

    def __embed_text(self, text):
        text_emb = self.model.encode([text])
        return text_emb

    def __strAll_docs(self) -> str:
        all_chunks = []
        for chunk in self.chunks_index:
            all_chunks.append(self.__read_chunk(chunk))
        return all_chunks

    def embed_chunks(self):
        all_chunkstr = self.__strAll_docs()
        chunks_embeddings = []
        print("chunking with tqdm")
        for chunk in tqdm.tqdm(all_chunkstr):
            chunks_embeddings.append(self.model.encode(chunk))

        self.chunk_embedding = chunks_embeddings
        with open(self.path_embeddings, "wb") as f:
            pickle.dump(self.chunk_embedding, f)
    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def load_or_create_embeddings(self):
        if os.path.exists(self.path_embeddings):
            with open(self.path_embeddings, "rb") as f:
                self.chunk_embedding = pickle.load(f)
        else:
            self.embed_chunks()

    
    def semantic_search(self, query, limit):
        query_embd = self.__embed_text(query)
        self.load_or_create_embeddings()
        map_chunks_score: List[Dict] = []

        for i, emb in enumerate(self.chunk_embedding):
            map_chunks_score.append(
                {
                    "chunk_indx": i,
                    "cosine_similarity": float(self.cosine_similarity(query_embd, emb))
                }
            )
        map_chunks_score = sorted(map_chunks_score, key=lambda item: item["cosine_similarity"], reverse=True)
        with open("data/processed/index.json", "r") as f:
            doc_chunks = json.load(f)
        semantic_map = {
                doc_chunks[r["chunk_indx"]]["id"]: {"chunk_indx": r["chunk_indx"], "cosine_similarity": r["cosine_similarity"]}
            for r in map_chunks_score[:limit]
        }
        return semantic_map

