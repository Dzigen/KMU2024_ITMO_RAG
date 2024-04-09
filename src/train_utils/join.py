import torch
from tqdm import tqdm
import numpy as np
import gc
import torch.nn as nn
import torch
from time import time
import json
import os
import torch.nn.functional as F

class JoinLoss:
    def __init__(self, r=1) -> None:
        self.temp = r

    def __call__(self, reader_topk_loss, reader_k_loss, retriever_k_scores):
        '''
        params:
            reader_topk_loss: 1
            reader_k_loss: BxN
            retriever_k_scores: BxN

        output:
            scores: 1
        '''

        retriever_part = torch.mean(torch.log(torch.sum(
            F.softmax(retriever_k_scores / self.temp, dim=1)*reader_k_loss, dim=1)))

        return reader_topk_loss + retriever_part
    
    def k_loss(self, reader_logits, labels):
        '''
        params:
            reader_logits: BxLxVOCAB_SIZE
            labels: BxL

        output:
            scores: B
        '''
        bsz, seq_len = reader_logits.shape[0], reader_logits.shape[1]

        return torch.mean(-torch.log(F.softmax(
            reader_logits.logits, dim=-1).gather(2,labels.view(bsz, seq_len, -1))).view(bsz, seq_len), dim=1)

#
def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])


def join_run(config, reader, retriever,  train_loader, eval_loader,
             train_func, evaluate_func, compare_score):

    print("Retriever-model parameters count: ",param_count(retriever.model))
    print("Reader-model parameters count: ",param_count(reader.model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        return
    os.mkdir(run_dir)
    logs_file_path = f'{config.base_dir}/{config.run_name}/logs.txt'
    retriever_bmodel_save_path = f"{config.base_dir}/{config.run_name}/retriever_bestmodel.pt"
    retriever_lmodel_save_path = f"{config.base_dir}/{config.run_name}/retriever_lastmodel.pt"
    reader_bmodel_save_path = f"{config.base_dir}/{config.run_name}/reader_bestmodel.pt"
    reader_lmodel_save_path = f"{config.base_dir}/{config.run_name}/reader_lastmodel.pt"

    print("Saving used nn-arch")
    with open(f"{config.base_dir}/{config.run_name}/used_reader_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(reader.model.__str__())
    with open(f"{config.base_dir}/{config.run_name}/used_retriever_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(retriever.model.__str__())

    print("Saving used config")
    with open(f"{run_dir}/used_config.json", 'w', encoding='utf-8') as fd:
        json.dump(config.__dict__, indent=2, fp=fd)

    print("Init train objectives")
    optimizer = torch.optim.AdamW([
        {'params': reader.model.parameters()}, 
        {'params': retriever.model.parameters()}], lr=config.lr)

    criterion = JoinLoss()

    ml_train = []
    ml_eval = []
    eval_scores = []
    best_score = 0

    print("===LEARNING START===")
    for i in range(config.epochs):

        print(f"Epoch {i+1} start:")
        train_s = time()
        train_losses = train_func(reader, retriever, train_loader, 
                                  optimizer, criterion, config)
        train_e = time()
        eval_losses, eval_metrics = evaluate_func(reader, retriever, eval_loader,
                                                   criterion, config)
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
        if best_score <= eval_scores[compare_score]:
            print("Update Best Model")
            if config.to_save:
                torch.save(retriever.model.state_dict(), retriever_bmodel_save_path)
                torch.save(reader.model.state_dict(), reader_bmodel_save_path)
            best_score = eval_scores[compare_score]

        # Save train/eval info to logs folder
        epoch_log = {
            'epoch': i+1, 'train_loss': ml_train[-1],
            'eval_losss': ml_eval[-1], 'scores': eval_scores[-1],
            'train_time': round(train_e - train_s, 5), 'eval_time': round(eval_e - train_e, 5)
            }
        with open(logs_file_path,'a',encoding='utf-8') as logfd:
            logfd.write(str(epoch_log) + '\n')

    print("===LEARNING END===")
    print("Best score: ", best_score)

    print("Save Last Model")
    if config.to_save:
        torch.save(retriever.model.state_dict(), retriever_lmodel_save_path)
        torch.save(reader.model.state_dict(), reader_lmodel_save_path)

    return ml_train, ml_eval, best_score, eval_scores

def join_evaluate(reader, retriever, eval_loader, criterion,
                config):
    pass

    # извлекаем k релевантных пассажей по запросу
    # конкатенируем запрос с пассажами
    # на их основе генерируем ответ 

    # выполняем оценку качества

def join_train(reader, retriever, train_loader, optimizer, criterion,
                config):
    pass


    # извлекаем k релевантных пассажей по query

    # заного прогоняем пассажи через retriever-часть
    # получаем скоры

    # конкатенируем пассажи и запросы
    # передаём в read модель
    # получем общий loss

    # подаём пассаж для каждлого запроса по отдельности
    # получем loss для каждого пассажа в отдельности
    # делаем stop gradioents

    # комбинируем скор из ретривера, общий лосс от ридера и разделённый лосс от ридера

