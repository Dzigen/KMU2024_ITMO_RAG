from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document
import pickle
import torc
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
import gc
import torch

import sys 
sys.path.insert(0, '/home/dzigen/Desktop/ITMO/ВКР/КМУ2024')

from retrievers.archs.colbert_model import ColBERT, ColBertTokenizer
gc.collect()

class BM25ColBertRetriever:
    def __init__(self, bm25_candidates=3, colbert_candidates=2, colbert_reddim=64, docs_bs=4) -> None:
        self.bm25_cands = bm25_candidates
        self.colbert_cands = colbert_candidates
        self.docs_bs = docs_bs

        self.colbert_model = ColBERT(reduced_dim=colbert_reddim)
        self.colbert_tokenizer = ColBertTokenizer
        self.tokenize = lambda x: self.colbert_tokenizer(
            x, max_length=512, truncation=True, 
            padding=True, return_tensors='pt')

    #
    def load_bm25_base(self, pickle_file):
        with open(pickle_file, 'rb') as bm25result_file:
            self.bm25_model = pickle.load(bm25result_file)

    #
    def load_colbert_model(self, weights_path):
        self.colbert_model.load_state_dict(torch.load(weights_path))

    #
    def texts2documents(self, texts):
        return [Document(page_content=txt, metadata={'tokenized': self.tokenize(txt)}) 
                for txt in texts]

    #
    def make_bm25_base(self, texts, save_pickle_file=None):
        documents = self.texts2documents(texts)
        self.bm25_model = BM25Retriever.from_documents(documents, 
                                                     k=self.bm25_cands)

        if save_pickle_file is not None:
            with open(save_pickle_file, 'wb') as bm25result_file:
                pickle.dump(self.bm25_model, bm25result_file)

    #
    def search(self, query, tokenized_query=None):
        # stage1
        bm25_docs, tokenized_docs = self.bm25_retrieve(query)

        # stage2
        if tokenized_query is None:
            tokenized_query = self.tokenize(query)
        colbert_docs, scores = self.colbert_retrieve(
            tokenized_query, bm25_docs, tokenized_docs)

        return colbert_docs, scores

    #
    def bm25_retrieve(self, query):
        relevant_documents = self.bm25_model.get_relevant_documents(query)
        text_docs = np.array([doc.page_content for doc in relevant_documents])
        tokenized_docs = [doc.metadata['tokenized'] for doc in relevant_documents]
        
        docs_dataset = DocDataset(tokenized_docs)
        docs_laoder = DataLoader(docs_dataset, batch_size=self.docs_bs, 
                                 collate_fn=custom_collate, shuffle=False)

        return text_docs, docs_laoder

    #
    def colbert_retrieve(self, tokenized_query, bm25_docs, docs_loader):
        all_scores = torch.tensor([], requires_grad=True)
        for doc_batch in docs_loader:
            print("doc_batch - ", doc_batch['input_ids'].shape)

            scores = self.colbert_model(tokenized_query['input_ids'], tokenized_query['attention_mask'],
                                    doc_batch['input_ids'], doc_batch['attention_mask'])
            all_scores = torch.cat((all_scores, scores),dim=1)

        flat_scores = all_scores.view(-1)
        _, indices = torch.sort(flat_scores, descending=True)
        relevant_ids = indices[:self.colbert_cands]

        print(bm25_docs)
        print(flat_scores)

        return bm25_docs[relevant_ids], flat_scores.take(relevant_ids)
    
class DocDataset(Dataset):
    def __init__(self, docs):
        self._data = docs

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]
        

def custom_collate(data):

    input_ids = torch.cat([item['input_ids'] for item in data], 0)
    attention_mask = torch.cat([item['attention_mask'] for item in data], 0)

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }