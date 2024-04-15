import sys
import os
import torch
import numpy as np
import random
from time import time
import json
from torch.utils.data import DataLoader
import re

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)

##############################

#ROOT_DIR = "/home/dzigen/Desktop/ITMO/ВКР/КМУ2024"
ROOT_DIR = "/home/ubuntu/KMU2024"
sys.path.insert(0, f"{ROOT_DIR}/src")

CONFIG_FILE_JSON = f'{ROOT_DIR}/learning_config.json'
DATA_DIR = f'{ROOT_DIR}/data'
BASES_DIR = f'{DATA_DIR}/bases'
LOGS_DIR = f'{ROOT_DIR}/logs'

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

print("Loading Reader-model...")
if learn_config.reader_type != '':

    if learn_config.reader_type == 'fid':
        reader = FiDReader(
            base_model=learn_config.base_reader_weights, 
            device=learn_config.device)

    else:
        print("Invalid 'reader_type'-value in config!")
        raise KeyError

    if learn_config.tuned_reader_weights != '':
        print("Loading Tuned weights...")
        reader.load_model(f"{LOGS_DIR}/{learn_config.tuned_reader_weights}")

    print("Loading metrics...")
    reader_metrics = ReaderMetrics(learn_config.base_dir)


    if (type(learn_config.reader_layers_toupdate) is str 
        and learn_config.reader_layers_toupdate == 'all'):
        print("All layers of reader-model will be updated!")
    elif type(learn_config.reader_layers_toupdate) is list:
        learn_config.reader_layers_toupdate = list(map(lambda x: "decoder.block."+str(x),learn_config.reader_layers_toupdate))
        learn_config.reader_layers_toupdate += ["decoder.final_layer_norm.weight", "lm_head.weight"]
        
        for name, param in reader.model.named_parameters():
            flag = False
            for layer in learn_config.reader_layers_toupdate:
                if len(re.findall(layer, name)):
                    flag = True
                    param.requires_grad = True
                    break
            if not flag:
                 param.requires_grad = False                    
    else:
        print("Invalid 'retriever_layers_toupdate'-value in config!")

else:
    print("Reader-model not used!")

##############################

print("Loadeing Retriever-model...")
if learn_config.retriever_type != '':
    if learn_config.retriever_type == 'bm25e5':
        retriever = BM25E5Retriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train',
            docs_bs=learn_config.retriever_docs_batch,
            bm25_candidates=learn_config.retriever_bm25_cands)
        
    elif learn_config.retriever_type == 'bm25colbert':
        retriever = BM25ColBertRetriever(
            k=learn_config.retrieved_cands, device=learn_config.device,
            mode='eval' if learn_config.retriever_frozen else 'train',
            docs_bs=learn_config.retriever_docs_batch, 
            bm25_candidates=learn_config.retriever_bm25_cands)

    elif learn_config.retriever_type == 'e5':
        retriever = E5Retriever(
            k=learn_config.retrieved_cands,
            device=learn_config.device)
        
    else:
        print("Invalid 'retriever_type'-value in config!")
        raise KeyError

    if learn_config.base != '':
        retriever.load_base(f"{BASES_DIR}/{learn_config.base}")

    if learn_config.tuned_retriever_weights != '':
        print("Loading Tuned weights...")
        retriever.load_model(f"{LOGS_DIR}/{learn_config.tuned_retriever_weights}")

    if (type(learn_config.retriever_layers_toupdate) is str 
        and learn_config.retriever_layers_toupdate == 'all'):
        print("All layers of retriever-model will be updated!")
    elif type(learn_config.retriever_layers_toupdate) is list:
        learn_config.retriever_layers_toupdate = list(map(lambda x: "encoder.layer."+str(x),learn_config.retriever_layers_toupdate))
        learn_config.retriever_layers_toupdate += ["encoder.pooler.dense.weight", "encoder.pooler.dense.bias","dim_reduce.weight","dim_reduce.bias"]
        for name, param in retriever.model.named_parameters():
            flag = False
            for layer in learn_config.retriever_layers_toupdate:
                if len(re.findall(layer, name)):
                    flag = True
                    param.requires_grad = True
                    break
            if not flag:
                 param.requires_grad = False                    
    else:
        print("Invalid 'retriever_layers_toupdate'-value in config!")


    print("Loading metrics...")
    retriever_metrics = RetrievalMetrics()

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
        reader.tokenize, learn_config.train_size)
    eval_dataset = CustomTriviaQADataset(
        f"{DATA_DIR}/TriviaQA",'eval', retriever.tokenize, 
        reader.tokenize, learn_config.eval_size)
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
    print('Invalid "train_dataset" value in config-file!')
    raise KeyError

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
            save_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        elif learn_config.train_type == 'unsupervised':
            pass

        else:
            print('Invalid "train_type" value in config-file!')
            raise KeyError
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
            save_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        elif learn_config.train_type == 'unsupervised':
            logs_fpath, _, _ = prepare_single_environment(learn_config, retriever)
            eval_losses, eval_metric = retriever_unsupervised_evaluate(
                learn_config, retriever, eval_loader, retriever_metrics, 0)
            save_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)

        else:
            print('Invalid "train_type" value in config-file!')
            raise KeyError

    else:
        single_run(learn_config, retriever,  train_loader, eval_loader, 
                retriever_supervised_train, retriever_supervised_evaluate, 
                retriever_metrics)

elif learn_config.run_type == 'join':

    if learn_config.eval_only:
        criterion = JoinLoss()
        logs_fpath, _, _, _, _ = prepare_join_environment(learn_config, reader, retriever)
        eval_losses, eval_metric = join_evaluate(
            learn_config, reader, retriever, eval_loader, 
            criterion, reader_metrics, retriever_metrics, 0)
        save_log(logs_fpath, 0, -1, round(np.mean(eval_losses),5), eval_metric, -1, -1)
    
    else:
        join_run(learn_config, reader, retriever, train_loader, eval_loader,
                join_train, join_evaluate, reader_metrics, retriever_metrics)

print("==RUN_END==")