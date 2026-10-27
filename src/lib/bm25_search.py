from src.lib.tokenizer import tokenizer
import os
import pickle
import json
import math
from typing import Dict, List


class KeyWordSearch:
    def __init__(self):
        self.inverse_index = {}
        with open("data/processed/index.json", "r") as f:
            self.doc_chunks = json.load(f)
        self.index_pickle_path = "data/processed/index.pkl"
        self.all_doc_len = self._calculate_doc_lenght()

    def __read_chunk(self, chunk):
        with open(chunk["path"], "r", encoding="utf-8", errors="replace") as f:
            start = chunk["first_character_index"]
            end = chunk["last_character_index"]
            chunk_string = f.read()[start:end]
        return chunk_string

    def _build_inverse_indx(self):
        print("build the inverse index...")
        for chunk in self.doc_chunks:
            chunk_str = self.__read_chunk(chunk)
            chunk_tokenized = tokenizer(chunk_str)
            for token in chunk_tokenized:
                if self.inverse_index.get(token) is None:
                    self.inverse_index.update(
                        {
                            token: {
                                chunk["id"]: 1
                            }
                        }
                        )
                elif self.inverse_index[token].get(chunk["id"]) is None:
                    self.inverse_index[token].update(
                        {
                            chunk["id"]: 1
                        }
                    )
                else:
                    self.inverse_index[token][chunk["id"]] += 1

        with open(self.index_pickle_path, "wb") as f:
            pickle.dump(self.inverse_index, f)

    def load_or_creatIndex(self):
        if os.path.exists(self.index_pickle_path):
            with open(self.index_pickle_path, "rb") as f:
                self.inverse_index = pickle.load(f)
        else:
            self._build_inverse_indx()

    def _calculate_doc_lenght(self) -> int:
        all_docs_lenght = 0
        for chunk in self.doc_chunks:
            end = chunk["last_character_index"]
            start = chunk["first_character_index"]
            all_docs_lenght += end - start
        return all_docs_lenght

    def bm25_tf(self, token, chunk_id, chunk_len):
        avrg = self.all_doc_len/len(self.doc_chunks)
        b = 0.75
        k = 1.2
        bm25tf = 0
        lenght_normalization = 1 - b + b * (chunk_len/avrg)
        token_index = self.inverse_index.get(token)
        if token_index and token_index.get(chunk_id):
            tf = token_index[chunk_id]
            bm25tf = (tf * (k + 1)) / (tf + k * lenght_normalization)
        return bm25tf
        # if all([self.inverse_index.get(token),
        #         self.inverse_index[token].get(chunk_id)]):
        #     tf = self.inverse_index[token][chunk_id]
        #     bm25tf = (tf * (k + 1)) / (tf + k * lenght_normalization)
        # return bm25tf

    def bm25_idf(self, token):
        bm25_idf = 0
        is_index = self.inverse_index.get(token)
        token_appearance = len(self.inverse_index[token]) if is_index else 0
        n_docs = len(self.doc_chunks)
        ratio = (n_docs - token_appearance + 0.5) / (token_appearance + 0.5)
        bm25_idf += math.log(1 + ratio, 2)
        return bm25_idf

    def bm25(self, query, limit) -> Dict:
        bm25_search: List[Dict] = []
        query_tokenized = tokenizer(query)
        self.load_or_creatIndex()

        for i, chunk in enumerate(self.doc_chunks):
            score = 0
            start = chunk["first_character_index"]
            end = chunk["last_character_index"]
            chunk_len = end - start
            chunk_id = chunk["id"]
            for token in query_tokenized:
                bm25_tf = self.bm25_tf(token, chunk_id, chunk_len)
                score += self.bm25_idf(token) * bm25_tf

            chunk_bm25 = {
                "chunk_indx": i,
                "bm25_score": score,
            }
            bm25_search.append(chunk_bm25)

        bm25_limit = sorted(bm25_search,
                            key=lambda item: item["bm25_score"],
                            reverse=True)
        map_bm25 = {
                    self.doc_chunks[r["chunk_indx"]]["id"]: {
                        "chunk_indx": r["chunk_indx"],
                        "bm25_score": r["bm25_score"]}
                    for r in bm25_limit[:limit]
            }

        return map_bm25
