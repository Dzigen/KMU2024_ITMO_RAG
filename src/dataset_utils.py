import torch
from datasets import Dataset
import pandas as pd

def reader_collate(data):
    cands = len(data)
    
    return {
        "ids": torch.cat([item['ids'] for item in data], 0).view(1, cands, -1), 
        "mask": torch.cat([item['mask'] for item in data], 0).view(1, cands, -1),
        "label": torch.cat([item['label'] for item in data], 0),
        "label_text": [item["label_text"] for item in data]
    }

# for reader only
class CustomSQuADDataset(Dataset):
    def __init__(self, data_dir, data_part, tokenizer):
        self.tokenizer = tokenizer
        data = pd.read_csv(f"{data_dir}/{data_part}.csv", sep=';')
        self._data = data[['title','question', 'context','answers','in_base_index']]

    def __len__(self,):
        return self._data.shape[0]

    def __getitem__(self, idx):
        
        query = self._data['question'][idx]
        title = self._data['title'][idx]
        context = self._data['context'][idx]
        
        input = f"TITLE: {title} PASSAGE: {context} QUESTION: {query}"
        tokenized_input = self.tokenizer([input])
        
        label = self._data['answers'][idx]
        tokenized_label = self.tokenizer([label])

        return {'ids': tokenized_input['input_ids'], 
                'mask': tokenized_input['attention_mask'], 
                'label': tokenized_label['input_ids'], 
                'label_text': label}

    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]