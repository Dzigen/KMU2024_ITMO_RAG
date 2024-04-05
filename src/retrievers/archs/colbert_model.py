import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_COLBERT_ENCODER = 'FacebookAI/roberta-base'
OUT_TOKEN_DIM = 768

ColBertTokenizer = AutoTokenizer.from_pretrained(BASE_COLBERT_ENCODER)

class ColBERT(nn.Module):
    def __init__(self, reduced_dim, base_model_path=BASE_COLBERT_ENCODER):
        super(ColBERT, self).__init__()

        self.query_encoder = AutoModel.from_pretrained(base_model_path)
        self.query_dim_reduce = nn.Linear(OUT_TOKEN_DIM, reduced_dim)

        self.document_encoder = AutoModel.from_pretrained(base_model_path)
        self.document_dim_reduce = nn.Linear(OUT_TOKEN_DIM, reduced_dim)

    def forward(self, q_ids, q_masks, d_ids, d_masks):
        '''
        params:
            q_ids: BxL
            q_masks: BxL
            d_ids: NxL
            d_masks: NxL
        
        output:
            scores: BxN
        '''
        print("=='colbert foward'-func")
        print("q_ids - ", q_ids.shape)
        print("q_masks - ", q_masks.shape)
        print("d_ids - ", d_ids.shape)
        print("d_masks - ", d_masks.shape)
        print()

        encoded_queries = self.q_enc(q_ids, q_masks)
        print("ecoded queries = ", encoded_queries.shape)

        encoded_documents = self.d_enc(d_ids, d_masks)
        print("encoded documents = ", encoded_documents.shape)

        scores = self.compute_scores(
            encoded_queries, q_masks, encoded_documents, d_masks)
        print("out scores = ", scores.shape)

        return scores


    def q_enc(self, q_ids, q_masks):
        '''
        params:
            q_ids: BxL
            q_mask: BxL

        output:
            query_embeddings: BxLxK
        '''
        print("=='q_enc'-function: ")
        print("q_ids - ", q_ids.shape)
        print("q_masks - ", q_masks.shape)
        print()

        print("out:")
        Q = self.query_encoder(q_ids, attention_mask=q_masks).last_hidden_state
        print("encoder -", Q.shape)
        Q = self.query_dim_reduce(Q)
        print("dim reduce -", Q.shape)

        return nn.functional.normalize(Q, dim=-1)

    def d_enc(self, d_ids, d_masks):
        '''
        params:
            d_ids: NxL
            d_mask: NxL

        output:
            document_embeddings: NxLxK
        '''
        print("=='d_enc'-function: ")
        print("d_ids - ", d_ids.shape)
        print("d_masks - ", d_masks.shape)
        print()

        print("out: ")
        D = self.document_encoder(d_ids, attention_mask=d_masks).last_hidden_state
        print("encoder - ", D.shape)
        D = self.document_dim_reduce(D)
        norm_d = nn.functional.normalize(D, dim=-1)
        print("dim reduce - ", D.shape)
        #NxLxK

        return norm_d

    def compute_scores(self, q_hidden, q_mask, d_hidden, d_mask):
        '''
        params:
            q_hidden: BxLxK
            q_mask: BxL
            d_hidden: NxLxK
            d_mask: NxL

        output:
            scores: BxN
        '''
        print("=='compute_score'-function: ")
        print('q_hidden - ', q_hidden.shape)
        print('q_mask - ', q_mask.shape)
        print('d_hidden - ', d_hidden.shape)
        print('d_mask - ', d_mask.shape)
        print()

        batch_scores = torch.tensor([], requires_grad=True)
        for i in range(q_hidden.shape[0]):
            C = F.cosine_similarity(q_hidden[i].unsqueeze(1).unsqueeze(1), 
                                    d_hidden, dim=-1)
            C = C.permute((1,0,2))
            print("cos similarity -",C.shape)

            d_masked_C = C.masked_fill(~d_mask.unsqueeze(1).bool(), -100)
            print("mask doc tokens - ",d_masked_C.shape)

            max_C = d_masked_C.max(dim=-1).values
            print("find max sim - ", max_C.shape)

            q_masked_C = max_C.masked_fill(~q_mask[i].unsqueeze(0).bool(), 0)
            print("mask query tokens - ",q_masked_C.shape)

            scores = q_masked_C.sum(dim=-1)
            print("sum scores - ",scores.shape)

            batch_scores = torch.cat((batch_scores, scores.unsqueeze(0)), dim=0)

        return batch_scores