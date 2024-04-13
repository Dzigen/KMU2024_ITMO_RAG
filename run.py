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

ROOT_DIR = '/home/dzigen/Desktop/ITMO/ВКР/КМУ2024'
#ROOT_DIR = '/home/ubuntu/KMU2024'
sys.path.insert(0, f"{ROOT_DIR}/src")

CONFIG_FILE_JSON = f'{ROOT_DIR}/learning_config.json'
DATA_DIR = f'{ROOT_DIR}/data'
BASES_DIR = f'{ROOT_DIR}/bases'

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
from src.dataset_utils import reader_collate, CustomSQuADDataset, CustomTriviaQADataset, retriever_collate, CustomMSMARCODataset, join_collate

##############################

# config
with open(CONFIG_FILE_JSON, 'r', encoding='utf-8') as fd:
    json_obj = json.loads(fd.read())
learn_config = RunConfig(**json_obj)

##############################

# metrics
retriever_metrics = RetrievalMetrics()
reader_metrics = ReaderMetrics()

##############################

print("Loading Reader-model...")
if learn_config.reader_type != '':

    if learn_config.reader_type == 'fid':
        reader = FiDReader(
            base_model=learn_config.base_reader_weights, 
            device=learn_config.device)

    else:
        assert "Invalid 'reader_type'-value in config!"

    if learn_config.tuned_reader_weights != '':
        print("Loading Tuned weights...")
        reader.load_model(learn_config.tuned_reader_weights)
else:
    print("Reader-model not used!")

##############################

print("Loadeing Retriever-model...")
if learn_config.retriever_type != '':
    if learn_config.retriever_type == 'bm25e5':
        retriever = BM25E5Retriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train')
        
    elif learn_config.retriever_type == 'bm25colbert':
        retriever = BM25ColBertRetriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train',
            docs_bs=learn_config.retriever_docs_batch)

    elif learn_config.retriever_type == 'e5':
        retriever = E5Retriever(
            k=learn_config.retrieved_cands,device=learn_config.device)
        
    else:
        assert "Invalid 'retriever_type'-value in config!"

    if learn_config.base != '':
        retriever.load_base(f"{BASES_DIR}/{learn_config.base}")

    if learn_config.tuned_retriever_weights != '':
        print("Loading Tuned weights...")
        retriever.load_model(learn_config.tuned_retriever_weights)

else:
    print("Retriever-model not used!")

##############################

print("Prepare Datasets and Dataloaders...")

if learn_config.dataset == 'squad':
    train_dataset = CustomSQuADDataset(
        f"{DATA_DIR}/SQuAD",'train', reader.tokenize,
        learn_config.reader_input_format, learn_config.train_size)
    eval_dataset = CustomSQuADDataset(
        f"{DATA_DIR}/SQuAD",'eval', reader.tokenize,
        learn_config.reader_input_format, learn_config.eval_size)
    custom_collate = reader_collate

elif learn_config.dataset == 'triviaqa':
    train_dataset = CustomTriviaQADataset(
        f"{DATA_DIR}/TriviaQA",'train', retriever.tokenize, 
        learn_config.train_size)
    eval_dataset = CustomTriviaQADataset(
        f"{DATA_DIR}/TriviaQA",'eval', retriever.tokenize, 
        learn_config.eval_size)
    custom_collate = join_collate

elif learn_config.dataset == 'msmarco':
    train_dataset = CustomMSMARCODataset(
        f"{DATA_DIR}/MSMARCO",'train', retriever.tokenize, 
        learn_config.train_size)
    eval_dataset = CustomMSMARCODataset(
        f"{DATA_DIR}/MSMARCO",'eval', retriever.tokenize, 
        learn_config.eval_size)
    custom_collate = retriever_collate

else:
    assert 'Invalid "train_dataset" value in config-file!'

train_loader = DataLoader(train_dataset, batch_size=learn_config.batch_size, 
                          collate_fn=custom_collate, shuffle=True, 
                          num_workers=2, drop_last=True)
eval_loader = DataLoader(eval_dataset, batch_size=learn_config.batch_size,
                         collate_fn=custom_collate, shuffle=False, num_workers=2, drop_last=True)

print(len(train_dataset), len(eval_dataset))

##############################

print("==RUN_START==")

if learn_config.run_type == 'reader':

    if learn_config.eval_only:
        if learn_config.train_type == 'supervised':
            logs_fpath, _, _ = prepare_single_environment(learn_config, reader)
            eval_losses, eval_metric = reader_supervised_evaluate(learn_config, reader, eval_loader, reader_metrics, 0)
            save_single_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        elif learn_config.train_type == 'unsupervised':
            pass

        else:
            assert 'Invalid "train_type" value in config-file!'
    else:
        single_run(learn_config, reader,  train_loader, eval_loader, 
                reader_supervised_train, reader_supervised_evaluate, 
                reader_metrics)

elif learn_config.run_type == 'retriever':

    if learn_config.eval_only:
        if learn_config.train_type == 'supervised':
            logs_fpath, _, _ = prepare_single_environment(learn_config, retriever)
            eval_losses, eval_metric = retriever_supervised_evaluate(
                learn_config, retriever, eval_loader, retriever_metrics, 0)
            save_single_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        elif learn_config.train_type == 'unsupervised':
            logs_fpath, _, _ = prepare_single_environment(learn_config, retriever)
            eval_losses, eval_metric = retriever_unsupervised_evaluate(
                learn_config, retriever, eval_loader, retriever_metrics, 0)
            save_single_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        else:
            assert 'Invalid "train_type" value in config-file!'

    else:
        single_run(learn_config, retriever,  train_loader, eval_loader, 
                retriever_supervised_train, retriever_supervised_evaluate, 
                retriever_metrics)

elif learn_config.run_type == 'join':

    if learn_config.eval_only:
        criterion = JoinLoss()
        eval_losses, eval_metric = join_evaluate(
            learn_config, reader, retriever, eval_loader, 
            criterion, reader_metrics, retriever_metrics, 0)
        
    else:
        join_run(learn_config, reader, retriever, train_loader, eval_loader,
                join_train, join_evaluate, reader_metrics, retriever_metrics)

print("==RUN_END==")