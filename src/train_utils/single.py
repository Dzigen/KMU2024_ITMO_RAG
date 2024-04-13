import torch
from tqdm import tqdm
import numpy as np
import gc
import torch.nn as nn
import torch
from time import time
import json
import os

#
def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

def prepare_single_environment(config, model_struct):
    print("Model parameters count: ",param_count(model_struct.model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/logs/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        return
    os.mkdir(run_dir)
    logs_file_path = f'{run_dir}/logs.txt'
    path_to_best_model_save = f"{run_dir}/bestmodel.pt"
    path_to_last_model_save = f"{run_dir}/lastmodel.pt"

    print("Saving used nn-arch")
    with open(f"{run_dir}/used_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(model_struct.model.__str__())

    print("Saving used config")
    with open(f"{run_dir}/used_config.json", 'w', encoding='utf-8') as fd:
        json.dump(config.__dict__, indent=2, fp=fd)

    return logs_file_path, path_to_best_model_save, path_to_last_model_save

#
def save_single_log(log_file, epoch, train_l, eval_l, eval_scores, train_t, eval_t):
    epoch_log = {
        'epoch': epoch, 'train_loss': train_l,
        'eval_losss': eval_l, 'scores': eval_scores,
        'train_time': train_t, 'eval_time': eval_t
        }
    with open(log_file,'a',encoding='utf-8') as logfd:
        logfd.write(str(epoch_log) + '\n')

#
def single_run(config, model_struct,  train_loader, eval_loader, 
               train_func, evaluate_func, metrics_obj):

    logs_file_path, path_to_best_model_save, path_to_last_model_save = prepare_single_environment(config, model_struct)

    print("Init train objectives")
    optimizer = torch.optim.AdamW(model_struct.model.parameters(), lr=config.lr)

    ml_train = []
    ml_eval = []
    eval_scores = []
    best_score = 0

    print("===LEARNING START===")
    for i in range(config.epochs):

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
        save_single_log(
            logs_file_path, i+1, ml_train[-1], ml_eval[-1], eval_scores[-1], 
            round(train_e - train_s, 5), round(eval_e - train_e, 5))

    print("===LEARNING END===")
    print("Best score: ", best_score)

    print("Save Last Model")
    if config.to_save:
         model_struct.save_model(path_to_last_model_save)

    return ml_train, ml_eval, best_score, eval_scores
