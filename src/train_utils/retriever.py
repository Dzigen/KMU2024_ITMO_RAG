import torch
import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

from src.config import RunConfig
from typing import Union, List, Tuple, Dict
from src.retrievers.bm25colbert import BM25ColBertRetriever
from src.retrievers.bm25e5 import BM25E5Retriever
from torch.utils.data import DataLoader
from src.metrics import RetrievalMetrics

#
def retriever_supervised_train(config: RunConfig, retriever: Union[BM25ColBertRetriever, BM25E5Retriever], 
                               loader: DataLoader, optimizer: object) -> List[float]:
    retriever.model.train()
    losses = []
    process = tqdm(loader)
    batch_keys = ['q_ids', 'q_mask', 'd_ids', 'd_mask']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        output = retriever.model(
            q_ids=batch['q_ids'], q_masks=batch['q_mask'],
            d_ids=batch['d_ids'], d_masks=batch['d_mask']
            )
        
        loss = retriever.loss(output)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses

#
def retriever_supervised_evaluate(config: RunConfig, retriever: Union[BM25ColBertRetriever, BM25E5Retriever], 
                                  loader: DataLoader, metrics_obj: RetrievalMetrics, epoch: int) -> Tuple[List[float], List[Dict[str,float]]]:
    scores = {
        'accuracy': [], 
    }
    losses = []

    process = tqdm(loader)
    batch_keys = ['q_ids', 'q_mask', 'd_ids', 'd_mask']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        output = retriever.model(
            q_ids=batch['q_ids'], q_masks=batch['q_mask'],
            d_ids=batch['d_ids'], d_masks=batch['d_mask']
            )
        
        loss = retriever.loss(output)
        predicted = output.argmax(dim=1).detach().cpu()
        target = torch.arange(0, batch['q_ids'].shape[0]).detach().cpu()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})
        scores['accuracy'].append(accuracy_score(target, predicted))

    scores = {
        'accuracy': round(np.mean(scores['accuracy']),5)
    } 

    return losses, scores

#
def retriever_unsupervised_train(config, retriever, loader, optimizer):
    pass

#
def retriever_unsupervised_evaluate(config, retriever, loader, metrics_obj, epoch):
    pass