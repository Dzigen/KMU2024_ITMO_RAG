import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_COLBERT_ENCODER = 'FacebookAI/roberta-base'
OUT_TOKEN_DIM = 768

ColBertTokenizer = AutoTokenizer.from_pretrained(BASE_COLBERT_ENCODER)

class ColBERT(nn.Module):
    def __init__(self, reduced_dim, base_model_path=BASE_COLBERT_ENCODER, device='cpu'):
        super(ColBERT, self).__init__()

        self.encoder = AutoModel.from_pretrained(base_model_path)
        self.dim_reduce = nn.Linear(OUT_TOKEN_DIM, reduced_dim)
        self.device = device

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
        # print("=='colbert foward'-func")
        # print("q_ids - ", q_ids.shape)
        # print("q_masks - ", q_masks.shape)
        # print("d_ids - ", d_ids.shape)
        # print("d_masks - ", d_masks.shape)
        # print()

        encoded_queries = self.enc(q_ids, q_masks)
        #print("ecoded queries = ", encoded_queries.shape)

        encoded_documents = self.enc(d_ids, d_masks)
        #print("encoded documents = ", encoded_documents.shape)

        scores = self.compute_scores(
            encoded_queries, q_masks, encoded_documents, d_masks)
        #print("out scores = ", scores.shape)

        return scores


    def enc(self, ids, masks):
        '''
        params:
            ids: BxL
            mask: BxL

        output:
            embeddings: BxLxK
        '''
        # print("=='q_enc'-function: ")
        # print("q_ids - ", q_ids.shape)
        # print("q_masks - ", q_masks.shape)
        # print()

        # print("out:")
        emds = self.encoder(ids, attention_mask=masks).last_hidden_state
        # print("encoder -", Q.shape)
        emds = self.dim_reduce(emds)
        # print("dim reduce -", Q.shape)

        return nn.functional.normalize(emds, dim=-1)

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
        # print("=='compute_score'-function: ")
        # print('q_hidden - ', q_hidden.shape)
        # print('q_mask - ', q_mask.shape)
        # print('d_hidden - ', d_hidden.shape)
        # print('d_mask - ', d_mask.shape)
        # print()

        batch_scores = torch.tensor([], requires_grad=True, device=self.device)
        for i in range(q_hidden.shape[0]):
            C = F.cosine_similarity(q_hidden[i].unsqueeze(1).unsqueeze(1), 
                                    d_hidden, dim=-1)
            C = C.permute((1,0,2))
            # print("cos similarity -",C.shape)

            d_masked_C = C.masked_fill(~d_mask.unsqueeze(1).bool(), -100)
            # print("mask doc tokens - ",d_masked_C.shape)

            max_C = d_masked_C.max(dim=-1).values
            # print("find max sim - ", max_C.shape)

            q_masked_C = max_C.masked_fill(~q_mask[i].unsqueeze(0).bool(), 0)
            # print("mask query tokens - ",q_masked_C.shape)

            scores = q_masked_C.sum(dim=-1)
            # print("sum scores - ",scores.shape)

            batch_scores = torch.cat((batch_scores, scores.unsqueeze(0)), dim=0)

        return batch_scores