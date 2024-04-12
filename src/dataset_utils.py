import torch
from datasets import Dataset
import pandas as pd
import ast

# for Join train
class CustomTriviaQADataset(Dataset):
    def __init__(self, path, part, tokenizer):
        self.tokenizer = tokenizer
        self._data = pd.read_csv(f"{path}/{part}.tsv", sep='\t')
        
    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self,idx):
        query = self._data['query'][idx]
        answer = self._data['answer'][idx]

        q_tokenized = self.tokenizer(query)
        a_tokenized = self.tokenizer(answer)

        return {
            'q_ids': q_tokenized['input_ids'],
            'q_mask': q_tokenized['attention_mask'],
            'q_text': [query],
            'label': a_tokenized['input_ids'],
            'label_text': [answer]
        }

    def __getitems__(self, idxs):
        queries = self._data.iloc[idxs,:]['query'].tolist()
        answers = self._data.iloc[idxs,:]['answer'].tolist()

        q_tokenized = self.tokenizer(queries)
        a_tokenized = self.tokenizer(answers)
        a_tokenized['input_ids'][a_tokenized['input_ids'] == 0] = -100

        return {
            'q_ids': q_tokenized['input_ids'],
            'q_mask': q_tokenized['attention_mask'],
            'q_text': queries,
            'label': a_tokenized['input_ids'],
            'label_text': answers
        }

def join_collate(data):
    return data

# for retriever only
class CustomMSMARCODataset(Dataset):
    def __init__(self, path, part, tokenizer):
        self.tokenizer = tokenizer
        self._data = pd.read_csv(f"{path}/{part}.tsv", sep='\t')
        
    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self,idx):
        query = self._data['query'][idx]
        document = self._data['document'][idx]

        q_tokenized = self.tokenizer(query)
        d_tokenized = self.tokenizer(document)

        return {
            'q_ids': q_tokenized['input_ids'],
            'q_mask': q_tokenized['attention_mask'],
            'd_ids': d_tokenized['input_ids'],
            'd_mask': d_tokenized['attention_mask']
        }

    def __getitems__(self, idxs):
        queries = self._data.iloc[idxs,:]['query'].tolist()
        documents = self._data.iloc[idxs,:]['document'].tolist()

        q_tokenized = self.tokenizer(queries)
        d_tokenized = self.tokenizer(documents)

        return {
            'q_ids': q_tokenized['input_ids'],
            'q_mask': q_tokenized['attention_mask'],
            'd_ids': d_tokenized['input_ids'],
            'd_mask': d_tokenized['attention_mask']
        }

def retriever_collate(data):
    return data


def reader_collate(data):
    bsz, cands = data['ids'].shape[0], 1
    
    return {
        "ids": data['ids'].view(bsz, cands, -1), 
        "mask": data['mask'].view(bsz, cands, -1),
        "label": data['label'],
        "label_text": data["label_text"]
    }

# for reader only
class CustomSQuADDataset(Dataset):
    def __init__(self, data_dir, data_part, tokenizer, input_format):
        self.tokenizer = tokenizer
        data = pd.read_csv(f"{data_dir}/{data_part}.csv", sep=';')
        self._data = data[['title','question', 'context','answers','in_base_index']]
        self._data['answers'] = self._data['answers'].apply(lambda v: ast.literal_eval(v)['text']) 
        self._data = self._data.drop(self._data[self._data['answers'].map(len) < 1].index).reset_index(drop=True)

        if data_part == 'train':
            self._data = self._data.iloc[:20000,:]

        self.input_format = input_format

    def __len__(self,):
        return self._data.shape[0]

    def __getitem__(self, idx):
        
        query = self._data['question'][idx]
        title = self._data['title'][idx]
        context = self._data['context'][idx]
        
        input = self.input_format.format(q=query,c=context)
        tokenized_input = self.tokenizer([input])
        
        label = self._data['answers'][idx][0]
        tokenized_label = self.tokenizer([label])

        tokenized_label['input_ids'][tokenized_label['input_ids'] == 0] = -100

        return {'ids': tokenized_input['input_ids'], 
                'mask': tokenized_input['attention_mask'], 
                'label': tokenized_label['input_ids'], 
                'label_text': label}

    def __getitems__(self, idxs):
        contexts = [self.input_format.format(
            q=self._data['question'][idx],c=self._data['context'][idx]) 
            for idx in idxs]
        
        labels = [self._data['answers'][idx][0] for idx in idxs]

        tokenized_contexts = self.tokenizer(contexts)
        tokenized_labels = self.tokenizer(labels)
        tokenized_labels['input_ids'][tokenized_labels['input_ids'] == 0] = -100

        return {'ids': tokenized_contexts['input_ids'], 
                'mask': tokenized_contexts['attention_mask'], 
                'label': tokenized_labels['input_ids'], 
                'label_text': labels}
