import torch
from tqdm import tqdm
import numpy as np
import gc
import torch.nn as nn
import torch
from time import time
import json
import os

from src.config import RunConfig
from typing import Union, List, Tuple, Dict
from src.readers.fid import FiDReader
from torch.utils.data import DataLoader
from src.metrics import ReaderMetrics

#
def param_count(model: object) -> int:
    all = sum([p.numel() for name, p in model.named_parameters()])
    trainable = sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

    return {"all": all, "trainable": trainable}

#
def prepare_single_environment(config: RunConfig, model_struct: object) -> Tuple[str,str,str]:
    print("Model parameters count: ",param_count(model_struct.model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/logs/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        raise KeyError
    
    os.mkdir(run_dir)
    logs_file_path = f'{run_dir}/logs.txt'
    path_to_best_model_save = f"{run_dir}/bestmodel.pt"
    path_to_last_model_save = f"{run_dir}/lastmodel.pt"

    print("Saving used nn-arch...")
    with open(f"{run_dir}/used_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(model_struct.model.__str__())

    print("Saving used config...")
    with open(f"{run_dir}/used_config.json", 'w', encoding='utf-8') as fd:
        json.dump(config.__dict__, indent=2, fp=fd)

    print("Saving grad info...")
    model_grad_info = ""
    for name, p in  model_struct.model.named_parameters():
        model_grad_info += f"{name} {p.requires_grad}\n"
    with open(f"{run_dir}/model_gradinfo.txt", 'w', encoding='utf-8') as fd:
        fd.write(model_grad_info)

    return logs_file_path, path_to_best_model_save, path_to_last_model_save

#
def save_log(log_file: str, epoch: int, train_l: float, eval_l: float, 
                    eval_scores: Dict[str,float], train_t: float, eval_t: float) -> None:
    epoch_log = {
        'epoch': epoch, 'train_loss': train_l,
        'eval_losss': eval_l, 'scores': eval_scores,
        'train_time': train_t, 'eval_time': eval_t
        }
    with open(log_file,'a',encoding='utf-8') as logfd:
        logfd.write(str(epoch_log) + '\n')

#
def single_run(config: RunConfig, model_struct: object,  train_loader: DataLoader, eval_loader: DataLoader, train_func: object, 
               evaluate_func: object, metrics_obj: Union[ReaderMetrics, ReaderMetrics]) -> Tuple[List[float], List[float], float, List[Dict[str, float]]]:

    logs_file_path, path_to_best_model_save, path_to_last_model_save = prepare_single_environment(config, model_struct)

    print("Init train objectives")
    optimizer = torch.optim.AdamW(model_struct.model.parameters(), lr=config.lr)

    ml_train = []
    ml_eval = []
    eval_scores = []
    best_score = 0

    print("===LEARNING START===")
    for i in range(config.epochs):
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Epoch {i+1} start:")
        train_s = time()
        train_losses = train_func(config, model_struct, train_loader, optimizer)
        
        torch.cuda.empty_cache()
        gc.collect()

        train_e = time()
        eval_losses, eval_metrics = evaluate_func(config, model_struct, eval_loader, metrics_obj, i)
        eval_e = time()
        
        torch.cuda.empty_cache()
        gc.collect()

        #
        ml_train.append(round(np.mean(train_losses),5))
        ml_eval.append(round(np.mean(eval_losses),5))
        eval_scores.append(eval_metrics)
        print(f"Epoch {i+1}: tain_loss - {round(ml_train[-1], 5)} | eval_loss - {round(ml_eval[-1],5)}")
        print(eval_scores[-1])

        #
        if best_score <= eval_scores[-1][config.base_score_compare]:
            print("Update Best Model")
            if config.to_save:
                model_struct.save_model(path_to_best_model_save)
            best_score = eval_scores[-1][config.base_score_compare]

        # Save train/eval info to logs folder
        save_log(
            logs_file_path, i+1, ml_train[-1], ml_eval[-1], eval_scores[-1], 
            round(train_e - train_s, 5), round(eval_e - train_e, 5))

    print("===LEARNING END===")
    print("Best score: ", best_score)

    print("Save Last Model")
    if config.to_save:
         model_struct.save_model(path_to_last_model_save)

    return ml_train, ml_eval, best_score, eval_scores
