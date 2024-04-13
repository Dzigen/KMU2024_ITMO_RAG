import torch
import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

#
def retriever_supervised_train(config, retriever, loader, optimizer):
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
def retriever_supervised_evaluate(config, retriever, loader, metrics_obj, epoch):
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