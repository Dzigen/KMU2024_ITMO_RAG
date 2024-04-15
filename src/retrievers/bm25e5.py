from langchain_community.retrievers import BM25Retriever
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
from src.dataset_utils import DocDataset, custom_collate

class BM25E5Retriever:
    def __init__(self, bm25_candidates=256, k=4, docs_bs=4, 
                 threshold=0, mode='eval', device='cpu') -> None:
        self.bm25_cands = bm25_candidates
        self.e5_cands = k
        self.docs_bs = docs_bs
        self.threshold = threshold
        self.mode = mode
        self.device = device

        print("Loading base E5-model...")
        self.model = E5()
        self.model.to(device)
        self.stage2_tokenizer = E5Tokenizer

        if mode == 'train':
            self.loss_objective = nn.CrossEntropyLoss()
            self.loss = lambda x: self.loss_objective(x, torch.arange(0, x.shape[0], device=self.device))

        self.tokenize = lambda x: self.stage2_tokenizer(
            x, return_tensors='pt', truncation=True, padding=True,
            add_special_tokens=True)

    #
    def load_base(self, pickle_file):
        print("Loading precomputed bm25-base...")
        with open(pickle_file, 'rb') as bm25result_file:
            self.bm25_model = pickle.load(bm25result_file)
        self.bm25_model.k = self.bm25_cands

    #
    def load_model(self, weights_path):
        print("Load tuned E5-model...")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)

    #
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

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
        #print("Retrieving documents with BM25...")
        bm25_docs, docs_loader, docs_metadata = self.bm25_retrieve(query)

        # stage2
        #print("Re-ranking documents with E5...")
        self.model.eval()
        if tokenized_query is None:
            tokenized_query = self.tokenize(query)
        e5_docs, scores, docs_metadata = self.e5_retrieve(
            tokenized_query, bm25_docs, docs_loader, docs_metadata)

        # stage3
        if self.mode == 'eval':
            #print("Filtering irrelevant document by threshold...")
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
        text_docs = [doc.page_content for doc in relevant_documents]
        tokenized_docs = self.tokenize(text_docs)
        metadata = [doc.metadata for doc in relevant_documents]

        docs_dataset = DocDataset(tokenized_docs)
        docs_laoder = DataLoader(docs_dataset, batch_size=self.docs_bs, 
                                 collate_fn=custom_collate, shuffle=False)

        return np.array(text_docs), docs_laoder, np.array(metadata)

    #
    def e5_retrieve(self, tokenized_query, bm25_docs, docs_loader, docs_metadata):
        flat_scores = torch.tensor([], requires_grad=True, device=self.device)

        for doc_batch in docs_loader:
            # print("doc_batch - ", doc_batch['input_ids'].shape)

            gc.collect()
            torch.cuda.empty_cache()

            doc_batch = {k: v.to(self.device) for k,v in doc_batch.items()}

            scores = self.model(tokenized_query['input_ids'], tokenized_query['attention_mask'],
                                    doc_batch['input_ids'], doc_batch['attention_mask'])
            
            flat_scores = torch.cat((flat_scores, scores.view(-1)), dim=-1)

        #print("BM25E5 flat scores: ", flat_scores.shape)

        _, indices = torch.sort(flat_scores, descending=True)
        best_ids = indices[:self.e5_cands]
        best_ids_cpu = best_ids.cpu()

        return bm25_docs[best_ids_cpu], flat_scores.take(best_ids), docs_metadata[best_ids_cpu]