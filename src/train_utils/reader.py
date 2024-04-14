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

def reader_supervised_evaluate(config: RunConfig, reader: Union[FiDReader], loader: DataLoader, 
                               metrics_obj: ReaderMetrics, epoch: int) -> Tuple[List[float], Dict[str, float]]:
    reader.model.eval()
    losses = []

    scores = {
        'bleu': [],
        'rouge': [],
        'meteor': [],
        'em': []
    }

    process = tqdm(loader)
    batch_keys = ['ids', 'mask', 'label']
    pred_answers = {}
    step = 0
    for batch in process:
        step += 1
        gc.collect()
        torch.cuda.empty_cache()

        device_b = {k: batch[k].to(config.device, non_blocking=True) for k in batch_keys}
        output = reader.model(input_ids=device_b['ids'], labels=device_b['label'],
                       attention_mask=device_b['mask'])

        losses.append(output.loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        output = reader.model.generate(input_ids=device_b['ids'], 
                                attention_mask=device_b['mask'], 
                                max_length=config.reader_gen_ml, 
                                eos_token_id=reader.tokenizer.eos_token_id)

        predicted = reader.tokenizer.batch_decode(output, skip_special_tokens=True)

        #print(predicted)     

        pred_answers[step] = {'gen':predicted, 'target': batch['label_text']}
        scores['bleu'].append(metrics_obj.bleu(predicted,batch['label_text']))
        scores['rouge'].append(metrics_obj.rouge(predicted,batch['label_text']))
        scores['meteor'].append(metrics_obj.meteor(predicted,batch['label_text']))
        scores['em'].append(metrics_obj.exact_match(predicted,batch['label_text']))

    scores = {
        'bleu': round(np.mean(scores['bleu']),5), 
        'rouge': round(np.mean(scores['rouge']),5),
        'meteor': round(np.mean(scores['meteor']),5),
        'em': round(np.mean(scores['em']),5)
        }

    print("Saving generated answers during evaluation...")
    with open(f"{config.base_dir}/logs/{config.run_name}/gen_answers_epoch{epoch}.json", 'w', encoding='utf-8') as fd:
        json.dump(pred_answers, indent=2, fp=fd)

    return losses, scores

def reader_supervised_train(config: RunConfig, reader: Union[FiDReader], 
                            loader: DataLoader, optimizer: object) -> List[float]:
    reader.model.train()
    losses = []
    process = tqdm(loader)
    batch_keys = ['ids', 'mask', 'label']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        #print(batch['ids'].shape, batch['mask'].shape, batch['label'].shape)

        output = reader.model(
            input_ids=batch['ids'], attention_mask=batch['mask'], 
            labels=batch['label'])

        output.loss.backward()
        optimizer.step()

        losses.append(output.loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses