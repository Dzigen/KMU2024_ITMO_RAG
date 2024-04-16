from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import numpy as np
import torch


from .archs.e5_model import E5_BASE_PATH, E5Tokenizer, E5

class E5Retriever:
    def __init__(self, k=4, device='cpu', threshold=0, mode='eval'):
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        self.cands = k
        self.threshold = threshold
        self.mode = mode
        self.device = device

        print("Loading query E5-model...")
        self.model = E5()
        self.model.to(device)

        print("Loading document E5-model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=E5_BASE_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        
        self.stage2_tokenizer = E5Tokenizer
        self.tokenize = lambda x: self.stage2_tokenizer(
            x, return_tensors='pt', truncation=True, padding=True,
            add_special_tokens=True)

    #   
    def texts2documents(self, texts, metadata):
        mdata = []
        for i in tqdm(range(len(texts))):
            tmp_m = {"text": texts[i]}
            if metadata is not None:
                tmp_m.update(metadata[i])
            mdata.append(tmp_m)

        return [Document(page_content="passage: "+txt, metadata=meta) 
                for txt, meta in tqdm(zip(texts, mdata))]

    #
    def make_base(self, texts, metadata, save_file=None):
        print("Converting texts with metadata to documents...")
        documents = self.texts2documents(texts, metadata)
        print("Indexing documents...")
        self.faiss = FAISS.from_documents(documents, self.embeddings, 
                                          distance_strategy='COSINE')

        if save_file is not None:
            self.faiss.save_local(save_file)

    #
    def load_model(self, weights_path):
        print("Loading tuned query E5-model...")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)

    #
    def load_base(self, file_path):
        print("Loading precomputed e5-base...")
        self.faiss = FAISS.load_local(file_path, self.model,
                                      allow_dangerous_deserialization=True,
                                      distance_strategy='COSINE')
    
    #
    def search(self, query, tokenized_query=None):

        # stage 1
        print("Retrieving documents with E5...")
        if tokenized_query is None:
            tokenized_query = self.tokenize("query: " + query)
            tokenized_query = {k: v.to(self.device) for k, v in tokenized_query.items()}

        query_hstates = self.model.enc(
            tokenized_query['input_ids'], 
            tokenized_query['attention_mask'])
        query_embd = self.model.average_pool(query_hstates, tokenized_query['attention_mask'])

        numpy_query = query_embd[0].detach().cpu().numpy()
        results = self.faiss.similarity_search_by_vector(
            numpy_query, k=self.cands) # 

        metadata = np.array([item.metadata for item in results])
        texts = np.array([meta["text"] for meta in metadata])

        doc_embeds = torch.tensor(self.embeddings.embed_documents(
            [item.page_content for item in results]), requires_grad=False, device=self.device)

        grad_scores = self.model.compute_scores(
            query_embd, None, doc_embeds, None, 
            q_pool=False, d_pool=False)

        return texts, grad_scores, metadata

    #
    def filter_by_score(self, scores):
        return torch.tensor([i for i, val in enumerate(scores) if val > self.threshold],
                             device=self.device)