import torch
import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

#
def retriever_supervised_train(retriever, loader, optimizer, config):
    retriever.model.train()
    losses = []
    process = tqdm(loader)
    batch_keys = ['q_input_ids', 'q_attention_mask', 'd_input_ids', 'd_attention_mask']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        output = retriever.model(
            q_ids=batch['q_input_ids'], q_masks=batch['q_attention_mask'],
            d_ids=batch['d_input_ids'], d_masks=batch['d_attention_mask']
            )
        
        loss = retriever.loss(output)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses

#
def retriever_supervised_evaluate(retriever, loader, config):
    scores = {
        'accuracy': [], 
    }
    losses = []

    process = tqdm(loader)
    batch_keys = ['q_input_ids', 'q_attention_mask', 'd_input_ids', 'd_attention_mask']
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        batch = {k: batch[k].to(config.device) for k in batch_keys}

        output = retriever.model(
            q_ids=batch['q_input_ids'], q_masks=batch['q_attention_mask'],
            d_ids=batch['d_input_ids'], d_masks=batch['d_attention_mask']
            )
        
        loss = retriever.loss(output)
        predicted = output.argmax(dim=1)
        target = torch.arange(0, batch['q_input_ids'][0])

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})
        scores['accuracy'].append(accuracy_score(target, predicted))

    scores = {
        'accuracy': round(np.mean(scores['accuracy']),5)
    } 

    return losses, scores

#
def retriever_unsupervised_train(retriever, loader, optimizer, config):
    pass

#
def retriever_unsupervised_evaluate(retriever, loader, config):
    pass