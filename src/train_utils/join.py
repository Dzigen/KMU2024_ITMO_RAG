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
            reader_logits: BxNxLxVOCAB_SIZE
            labels: BxL

        output:
            scores: BxN
        '''
        bsz, k, seq_len = reader_logits.shape[0], reader_logits.shape[1], reader_logits.shape[2]

        return torch.mean(-torch.log(F.softmax(
            reader_logits.logits, dim=-1).gather(3,labels.view(bsz, 1, seq_len, -1))).view(bsz, k, seq_len), dim=2)

#
def param_count(model):
    return sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])


def join_run(config, reader, retriever,  train_loader, eval_loader,
             train_func, evaluate_func, reader_metrics, retriever_metrics):

    print("Retriever-model parameters count: ",param_count(retriever.model))
    print("Reader-model parameters count: ",param_count(reader.model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/logs/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        return
    os.mkdir(run_dir)
    logs_file_path = f'{run_dir}/logs.txt'
    retriever_bmodel_save_path = f"{run_dir}/retriever_bestmodel.pt"
    retriever_lmodel_save_path = f"{run_dir}/retriever_lastmodel.pt"
    reader_bmodel_save_path = f"{run_dir}/reader_bestmodel.pt"
    reader_lmodel_save_path = f"{run_dir}/reader_lastmodel.pt"

    print("Saving used nn-arch")
    with open(f"{run_dir}/used_reader_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(reader.model.__str__())
    with open(f"{run_dir}/used_retriever_arch.txt", 'w', encoding='utf-8') as fd:
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
        train_losses = train_func(config, reader, retriever, train_loader, 
                                  optimizer, criterion)
        train_e = time()
        eval_losses, eval_metrics = evaluate_func(config, reader, retriever, 
                                                  eval_loader,  criterion, 
                                                  reader_metrics, retriever_metrics, i)
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
                torch.save(retriever.model.state_dict(), retriever_bmodel_save_path)
                torch.save(reader.model.state_dict(), reader_bmodel_save_path)
            best_score = eval_scores[-1][config.base_score_compare]

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

def join_evaluate(config, reader, retriever,  loader,  criterion, 
                  reader_metrics, retriever_metrics, epoch):
    reader.model.eval()
    retriever.model.eval()
    
    scores = {
        'reader': {
            'bleu': [],'rouge': [],
            'meteor': [],'em': []
        },
        'retriever': {
            'MRR': [],
            'mAP': []
        }
    }
    losses = []

    pred_answers = []
    step = 0
    process = tqdm(loader)
    for batch in process:
        step += 1
        gc.collect()
        torch.cuda.empty_cache()

        q_ids = batch['q_ids'].to(config.device, non_blocking=True)
        q_masks = batch['q_mask'].to(config.device, non_blocking=True)

        # retriever-part
        prep_txts = []
        doc_bsz = config.retriever_docs_batch
        cands_k = config.retrieved_cands
        cands_scores = torch.tensor([], requires_grad=True)
        for i in range(len(batch['q_text'])):
            texts, k_scores, metadata = retriever.search(
                batch['q_text'][i], {'input_ids': q_ids[i].view(1,-1), 
                                     'attention_mask':q_masks[i].view(1,-1)})

            if 'relevant_d_ids' in batch.keys():
                predicted_d_ids = [meta['in_base_index'] for meta in metadata]

                scores['retriever']['MRR'].append(retriever_metrics.mAP(
                    predicted_d_ids,batch['relevant_d_ids'][i]))
                scores['retriever']['mAP'].append(retriever_metrics.MRR(
                    predicted_d_ids,batch['relevant_d_ids'][i]))

            cands_scores = torch.cat((cands_scores, k_scores), dim=0)
            prep_txts += list(map(lambda t: config.reader_input_format.format(q=batch['q_text'][i],c=t), texts))
        
        # reader-part
        tokenized_txts = reader.tokenize(prep_txts)
        tokenized_txts = {k: v.to(config.device) for k, v in tokenized_txts.items()}

        output = reader.model(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz*cands_k, 1, -1), 
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz*cands_k, 1, -1))

        print("k_states: ", output.last_hidden_states.shape)
        reader_k_loss = criterion.k_loss(output.last_hidden_states, batch['label'])

        output = reader.model(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
            labels=batch['label'], 
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
        
        print("topk_states: ", output.last_hidden_states.shape)
        reader_topk_loss = output.loss.item()

        loss = criterion(reader_topk_loss, reader_k_loss, cands_scores.view(doc_bsz, cands_k))

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        output = reader.model.generate(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1),
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1), max_length=64)   

        predicted = reader.tokenizer.batch_decode(output, skip_special_tokens=True)

        pred_answers[step] = {'gen':predicted, 'target': batch['label_text']}
        scores['reader']['bleu'].append(reader_metrics.bleu(predicted,[batch['label_text'][i]]))
        scores['reader']['rouge'].append(reader_metrics.rouge(predicted,[batch['label_text'][i]]))
        scores['reader']['meteor'].append(reader_metrics.meteor(predicted,[batch['label_text'][i]]))
        scores['reader']['em'].append(reader_metrics.exact_match(predicted,[batch['label_text'][i]]))

    print("Saving generated answers during evaluation...")
    with open(f"{config.base_dir}/logs/{config.run_name}/gen_answers_epoch{epoch}.json", 'w', encoding='utf-8') as fd:
        json.dump(pred_answers, indent=2, fp=fd)

    return losses, scores

def join_train(config, reader, retriever, loader, 
               optimizer, criterion):
    reader.model.train()
    retriever.model.train()

    losses = []
    process = tqdm(loader)
    for batch in process:
        gc.collect()
        torch.cuda.empty_cache()

        q_ids = batch['q_ids'].to(config.device, non_blocking=True)
        q_masks = batch['q_mask'].to(config.device, non_blocking=True)

        # retriever-part
        prep_txts = []
        doc_bsz = config.retriever_docs_batch
        cands_k = config.retrieved_cands
        cands_scores = torch.tensor([], requires_grad=True)
        for i in range(len(batch['q_text'])):
            texts, k_scores, _ = retriever.search(
                batch['q_text'][i], {'input_ids': q_ids[i].unsqueeze(0), 
                                     'attention_mask':q_masks[i].unsqueeze(0)})

            cands_scores = torch.cat((cands_scores, k_scores), dim=0)
            prep_txts += list(map(lambda t: config.reader_input_format.format(q=batch['q_text'][i],c=t), texts))
        
        # reader-part
        tokenized_txts = reader.tokenize(prep_txts)
        tokenized_txts = {k: v.to(config.device) for k, v in tokenized_txts.items()}

        output = reader.model(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz*cands_k, 1, -1), 
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz*cands_k, 1, -1))

        print("k_states: ", output.last_hidden_states.shape)
        reader_k_loss = criterion.k_loss(output.last_hidden_states, batch['label'])

        output = reader.model(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
            labels=batch['label'], 
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
        
        print("topk_states: ", output.last_hidden_states.shape)
        reader_topk_loss = output.loss

        loss = criterion(reader_topk_loss, reader_k_loss, cands_scores.view(doc_bsz, cands_k))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

    return losses

