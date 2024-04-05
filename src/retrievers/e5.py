import pickle
import torc
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import gc
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import sys 
sys.path.insert(0, '/home/dzigen/Desktop/ITMO/ВКР/КМУ2024')
gc.collect()



# faiss from langchain

from archs.e5_model import E5Tokenizer, E5_BASE_PATH

class E5Retriever:
    def __init__(self, texts, k):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.cands =k

        self.embeddings = HuggingFaceEmbeddings(
            model_name=E5_BASE_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        
        self.e5_tokenizer = E5Tokenizer

        self.tokenize = lambda x: self.e5_tokenizer(
            x, max_length=512, truncation=True, 
            padding=True, return_tensors='pt')

        documents = self.texts2documents(texts)

        self.faiss = FAISS.from_documents(documents, self.embeddings)

    #
    def texts2documents(self, texts):
        return [Document(page_content=txt, metadata={'tokenized': self.tokenize(txt)}) 
                for txt in texts]
    
    #
    def search(self, query):
        results = self.faiss.similarity_search_with_score(query, k=self.cands)

        scores = list(map(lambda item: item[1], results))
        texts = list(map(lambda item: item[0].page_content, results))
        tokenized_texts = list(map(lambda item: item[0].metadata['tokenized'], results))

        return scores, texts, tokenized_texts