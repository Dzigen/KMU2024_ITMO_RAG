
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

E5_BASE_PATH = "intfloat/multilingual-e5-base"

E5Tokenizer = AutoTokenizer.from_pretrained(E5_BASE_PATH)

class E5(nn.Module):
    def __init__(self, base_model_path=E5_BASE_PATH):
        super(E5, self).__init__()

        self.encoder = AutoModel.from_pretrained(base_model_path)

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

        encoded_queries = self.enc(q_ids, q_masks)
        print("ecoded queries = ", encoded_queries.shape)

        encoded_documents = self.enc(d_ids, d_masks)
        print("encoded documents = ", encoded_documents.shape)

        scores = self.compute_scores(
            encoded_queries, q_masks, encoded_documents, d_masks)
        print("out scores = ", scores.shape)

        return scores

    def enc(self, ids, masks):
        '''
        params:
            ids: BxL | NxL
            mask: BxL | NxL

        output:
            embeddings: BxLxK | NxLxK
        '''
        print("=='enc'-function: ")
        print("ids - ", ids.shape)
        print("masks - ", masks.shape)
        print()

        print("out:")
        last_hidden = self.encoder(ids, attention_mask=masks).last_hidden_state
        print("encoder -", last_hidden.shape)
        norm_hidden = nn.functional.normalize(last_hidden, dim=-1)

        return norm_hidden

    def average_pool(self, last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            print(last_hidden.shape)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def compute_scores(self, q_hidden, q_masks, d_hidden, d_masks, q_pool=True, d_pool=True):
        '''
        params:
            q_hidden: BxLxK | BxK
            q_mask: BxL
            d_hidden: NxLxK | NxK
            d_mask: NxL
            q_pool: True | False
            d_pool: True | False

        output:
            scores: BxN
        '''
        print("=='compute_score'-function: ")
        print('q_hidden - ', q_hidden.shape)
        print('q_mask - ', q_masks.shape)
        print('d_hidden - ', d_hidden.shape)
        print('d_mask - ', d_masks.shape)
        print()
        
        q_embds = self.average_pool(q_hidden, q_masks) if q_pool else q_hidden
        d_embds = self.average_pool(d_hidden, d_masks) if d_pool else d_hidden

        print("avg hidden: ")
        print("q - ", q_embds.shape)
        print("d - ", d_embds.shape)

        scores = F.cosine_similarity(q_embds.unsqueeze(1), d_embds, dim=-1)
        
        return scores