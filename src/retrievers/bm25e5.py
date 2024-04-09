from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document
import pickle
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
import gc
import torch
from tqdm import tqdm
import torch.nn as nn

from .archs.e5_model import E5, E5Tokenizer

class BM25E5Retriever:
    def __init__(self, bm25_candidates=256, e5_candidates=4, docs_bs=4, 
                 threshold=0, mode='eval') -> None:
        self.bm25_cands = bm25_candidates
        self.e5_cands = e5_candidates
        self.docs_bs = docs_bs
        self.threshold = threshold
        self.mode = mode

        print("Loading base E5-model...")
        self.model = E5()
        self.stage2_tokenizer = E5Tokenizer

        if mode == 'train':
            self.loss_objective = nn.CrossEntropyLoss()
            self.loss = lambda x: self.loss_objective(x, torch.arange(0, x.shape[0]))

        self.tokenize = lambda x: self.stage2_tokenizer(
            x, max_length=512, truncation=True, 
            padding='max_length', return_tensors='pt')

    #
    def load_bm25_base(self, pickle_file):
        print("Loading precomputed base...")
        with open(pickle_file, 'rb') as bm25result_file:
            self.bm25_model = pickle.load(bm25result_file)

    #
    def load_e5_model(self, weights_path):
        print("Load tuned E5-model...")
        self.model.load_state_dict(torch.load(weights_path))

    #
    def texts2documents(self, texts, metadata=[]):
        mdata = []
        for i in tqdm(range(len(texts))):
            tmp_m = {'e5_tokenized': self.tokenize("passage: " + texts[i])}
            if metadata is not None:
                tmp_m.update(metadata[i])
            mdata.append(tmp_m)

        return [Document(page_content=txt, metadata=meta) 
                for txt, meta in zip(texts, mdata)]

    #
    def make_bm25_base(self, texts, metadata=[], save_pickle_file=None):
        print("Converting texts with metadata to documents...")
        documents = self.texts2documents(texts, metadata)
        print("Indexing documents...")
        self.bm25_model = BM25Retriever.from_documents(documents, 
                                                     k=self.bm25_cands)

        if save_pickle_file is not None:
            with open(save_pickle_file, 'wb') as bm25result_file:
                pickle.dump(self.bm25_model, bm25result_file)

    #
    def search(self, query, tokenized_query=None):
        # stage1
        print("Retrieving documents with BM25...")
        bm25_docs, docs_loader, docs_metadata = self.bm25_retrieve(query)

        # stage2
        print("Re-ranking documents with E5...")
        self.model.eval()
        if tokenized_query is None:
            tokenized_query = self.tokenize(query)
        e5_docs, scores, docs_metadata = self.e5_retrieve(
            tokenized_query, bm25_docs, docs_loader, docs_metadata)

        # stage3
        if self.mode == 'eval':
            print("Filtering irrelevant document by threshold...")
            filtered_indexes = self.filter_by_score(scores)

            return e5_docs[filtered_indexes], scores.take(filtered_indexes), docs_metadata[filtered_indexes]
    
        else:
            return e5_docs, scores, docs_metadata


    #
    def filter_by_score(self, scores):
        return torch.tensor([i for i, val in enumerate(scores) if val > self.threshold])

    #
    def bm25_retrieve(self, query):
        relevant_documents = self.bm25_model.get_relevant_documents(query)
        text_docs = np.array([doc.page_content for doc in relevant_documents])
        tokenized_docs = [doc.metadata['e5_tokenized'] for doc in relevant_documents]
        metadata = np.array([doc.metadata for doc in relevant_documents])

        docs_dataset = DocDataset(tokenized_docs)
        docs_laoder = DataLoader(docs_dataset, batch_size=self.docs_bs, 
                                 collate_fn=custom_collate, shuffle=False)

        return text_docs, docs_laoder, metadata

    #
    def e5_retrieve(self, tokenized_query, bm25_docs, docs_loader, docs_metadata):
        flat_scores = torch.tensor([], requires_grad=False)

        for doc_batch in tqdm(docs_loader):
            # print("doc_batch - ", doc_batch['input_ids'].shape)

            gc.collect()
            torch.cuda.empty_cache()

            scores = self.model(tokenized_query['input_ids'], tokenized_query['attention_mask'],
                                    doc_batch['input_ids'], doc_batch['attention_mask'])
            
            flat_scores = torch.cat((flat_scores, scores.detach().view(-1)), dim=-1)

        _, indices = torch.sort(flat_scores, descending=True)
        best_ids = indices[:self.e5_cands]

        return bm25_docs[best_ids], flat_scores.take(best_ids), docs_metadata[best_ids]
    
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