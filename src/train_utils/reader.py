import torch
from tqdm import tqdm
import numpy as np
import gc
import torch.nn as nn
import torch
from time import time
import json
import os

def reader_supervised_evaluate(config, reader, loader, metrics_obj):
    reader.model.eval()
    losses = []
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    em_scores = []

    process = tqdm(loader)
    batch_keys = ['ids', 'mask', 'label']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        batch = {k: batch[k].to(config.device, non_blocking=True) for k in batch_keys}
        output = reader.model(input_ids=batch['ids'], labels=batch['label'],
                       attention_mask=batch['mask'])

        losses.append(output.loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        output = reader.model.generate(input_ids=batch['ids'], 
                                attention_mask=batch['mask'], 
                                max_length=128)     

        predicted = reader.tokenizer.batch_decode(output, skip_special_tokens=True)

        bleu_scores.append(metrics_obj.bleu(predicted,batch['label_text']))
        rouge_scores.append(metrics_obj.rouge(predicted,batch['label_text']))
        meteor_scores.append(metrics_obj.meteor(predicted,batch['label_text']))
        em_scores.append(metrics_obj.exact_match(predicted,batch['label_text']))

    scores = {
        'bleu': round(np.mean(bleu_scores),5), 
        'rouge': round(np.mean(rouge_scores),5),
        'meteor': round(np.mean(meteor_scores),5),
        'em': round(np.mean(em_scores),5)}

    return losses, scores

def reader_supervised_train(config, reader, loader, optimizer):
    reader.model.train()
    losses = []
    process = tqdm(loader)
    batch_keys = ['ids', 'mask', 'label']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        output = reader.model(**batch)

        output.loss.backward()
        optimizer.step()

        losses.append(output.loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses