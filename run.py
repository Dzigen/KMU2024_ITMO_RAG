import sys
import os
import torch
import numpy as np
import random
from time import time
import json
from torch.utils.data import DataLoader

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)

##############################

CONFIG_FILE_JSON = f'./learning_config.json'

# config
with open(CONFIG_FILE_JSON, 'r', encoding='utf-8') as fd:
    json_obj = json.loads(fd.read())
learn_config = RunConfig(**json_obj)

LOGS_DIR = f'{learn_config.base_dir}/logs'
DATA_DIR = f'{learn_config.base_dir}/data'
BASES_DIR = f'{learn_config.base_dir}/bases'

##############################

sys.path.insert(0, f"{learn_config.base_dir}/src")

from src.config import RunConfig
from src.train_utils.single import *
from src.train_utils.join  import *
from src.train_utils.reader import *
from src.readers.fid import FiDReader
from src.retrievers.bm25colbert import BM25ColBertRetriever
from src.retrievers.bm25e5 import BM25E5Retriever
from src.retrievers.e5 import E5Retriever
from src.train_utils.retriever import *
from src.metrics import ReaderMetrics, RetrievalMetrics
from src.dataset_utils import reader_collate, CustomSQuADDataset

##############################

# metrics
retriever_metrics = RetrievalMetrics()
reader_metrics = ReaderMetrics()

##############################

print("Loadeing Reader-model...")
if learn_config.reader_type != '':

    if learn_config.reader_type == 'fid':
        reader = FiDReader(device=learn_config.device)

    else:
        assert "Invalid 'reader_type'-value in config!"

    if learn_config.tuned_reader_weights != '':
        print("Loading Tuned weights...")
        reader.load_model(learn_config.tuned_reader_weights)
else:
    print("Reader-model not used!")

##############################

print("Loadeing Retriever-model...")
if learn_config.retrieved_type != '':
    if learn_config.retrieved_type == 'bm25e5':
        retriever = BM25E5Retriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train')
        
    elif learn_config.retrieved_type == 'bm25colbert':
        retriever = BM25ColBertRetriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train',
            docs_bs=learn_config.retriever_docs_batch)

    elif learn_config.retrieved_type == 'e5':
        retriever = E5Retriever(
            k=learn_config.retrieved_cands,device=learn_config.device)
        
    else:
        assert "Invalid 'retrieved_type'-value in config!"

    retriever.load_base(f"{BASES_DIR}/{learn_config.base}")

    if learn_config.tuned_retriever_weights != '':
        print("Loading Tuned weights...")
        retriever.load_model(learn_config.tuned_retriever_weights)

else:
    print("Retriever-model not used!")

##############################

print("Prepare Datasets and Dataloaders...")

if learn_config.train_dataset == 'squad':
    train_dataset = CustomSQuADDataset(f"{DATA_DIR}/squad",'train')
    eval_dataset = CustomSQuADDataset(f"{DATA_DIR}/squad",'eval')
    custom_collate = reader_collate
else:
    assert 'Invalid "train_dataset" value in config-file!'

train_loader = DataLoader(train_dataset, batch_size=learn_config.batch_size, 
                          collate_fn=custom_collate, shuffle=True, 
                          num_workers=2, drop_last=True)
eval_loader = DataLoader(train_dataset, batch_size=learn_config.batch_size,
                         collate_fn=custom_collate, shuffle=False, num_workers=2)

##############################

if learn_config.run_type == 'reader':

    single_run(learn_config, reader,  train_loader, eval_loader, 
               reader_supervised_train, reader_supervised_evaluate, 
               reader_metrics)

elif learn_config.run_type == 'retriever':

    single_run(learn_config, retriever,  train_loader, eval_loader, 
               retriever_supervised_train, retriever_supervised_evaluate, 
               retriever_metrics)

elif learn_config.run_type == 'join':

    join_run(learn_config, reader, retriever,  train_loader, eval_loader,
             join_train, join_evaluate, learn_config.base_score_compare)
