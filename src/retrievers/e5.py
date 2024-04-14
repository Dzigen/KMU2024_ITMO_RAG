from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import numpy as np
import torch


from .archs.e5_model import E5_BASE_PATH, E5Tokenizer

class E5Retriever:
    def __init__(self, k=4, device='cpu', threshold=0, mode='eval'):
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        self.cands = k
        self.threshold = threshold
        self.mode = mode

        print("Loading E5-model...")
        self.model = HuggingFaceEmbeddings(
            model_name=E5_BASE_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        
        self.stage2_tokenizer = E5Tokenizer
        self.tokenize = lambda x: self.stage2_tokenizer(
            x, return_tensors='pt', truncation=True, padding=True,
            add_special_tokens=True)

    #
    def make_base(self, texts, metadata, save_file=None):
        print("Converting texts with metadata to documents...")
        documents = self.texts2documents(texts, metadata)
        print("Indexing documents...")
        self.faiss = FAISS.from_documents(documents, self.model)

        if save_file is not None:
            self.faiss.save_local(save_file)

    #
    def load_base(self, file_path):
        print("Loading precomputed e5-base...")
        self.faiss = FAISS.load_local(file_path, self.model,
                                      allow_dangerous_deserialization=True)

    #   
    def texts2documents(self, texts, metadata):
        return [Document(page_content="passage: "+txt, metadata=meta) 
                for txt, meta in tqdm(zip(texts, metadata))]
    
    #
    def search(self, query, tokenized_query=None):
        # stage 1
        print("Retrieving documents with E5...")
        results = self.faiss.similarity_search_with_score('query: '+query, k=self.cands)

        scores = torch.tensor([item[1] for item in results])
        texts = np.array([item[0].page_content[9:] for item in results])
        metadata = np.array([item[0].metadata for item in results])

        # stage 2
        if self.mode == 'eval':
            print("Filtering irrelevant document by threshold...")
            filtered_indexes = self.filter_by_score(scores)

            return texts[filtered_indexes], scores.take(filtered_indexes), metadata[filtered_indexes]

        else:
            return texts, scores, metadata

    #
    def filter_by_score(self, scores):
        return torch.tensor([i for i, val in enumerate(scores) if val > self.threshold])