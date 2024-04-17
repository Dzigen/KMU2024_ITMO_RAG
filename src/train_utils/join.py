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

from src.config import RunConfig
from typing import Union, List, Tuple, Dict
from src.retrievers.bm25colbert import BM25ColBertRetriever
from src.retrievers.bm25e5 import BM25E5Retriever
from src.retrievers.e5 import E5Retriever
from src.readers.fid import FiDReader
from torch.utils.data import DataLoader
from src.metrics import RetrievalMetrics, ReaderMetrics
from src.train_utils.single import save_log

#
class JoinLoss:
    def __init__(self, r=1) -> None:
        self.temp = r
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')

    def __call__(self, reader_topk_loss, reader_k_loss, retriever_k_scores):
        '''
        params:
            reader_topk_loss: 1
            reader_k_loss: BxN
            retriever_k_scores: BxN

        output:
            scores: 1
        '''
        #print("JOINLOSS-method=")
        #print("topk: ", reader_topk_loss)
        #print("k_loss:", reader_k_loss.shape)
        #print("retriever scores: ", retriever_k_scores.shape)

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
        #print("KLOSS-method=")
        #print("reader_logits: ", reader_logits.shape)
        #print("labels:", labels.shape)

        bsz, k, seq_len = reader_logits.shape[0], reader_logits.shape[1], reader_logits.shape[2]

        out = self.cross_entropy(
            reader_logits.view(-1, reader_logits.size(-1)), 
            labels.repeat(1,k).view(bsz*k, seq_len).view(-1))
        
        #print("ce_out: ", out.shape)

        return torch.mean(out.view(bsz, k, seq_len),2)

#
def prepare_join_environment(config: RunConfig, reader: object, retriever: object) -> Tuple[str,str,str]:
    
    print("Reader-model parameters count: ",param_count(reader.model))
    if not config.retriever_frozen:
        print("Retriever-model parameters count: ",param_count(retriever.model))

    print("Init folder to save")
    run_dir = f"{config.base_dir}/logs/{config.run_name}"
    if os.path.isdir(run_dir):
        print("Error: Директория существует")
        raise KeyError
    
    os.mkdir(run_dir)
    logs_file_path = f'{run_dir}/logs.txt'
    retbmodel_spath = f"{run_dir}/retriever_bestmodel.pt"
    retlmodel_spath = f"{run_dir}/retriever_lastmodel.pt"
    readbmodel_spath = f"{run_dir}/reader_bestmodel.pt"
    readlmodel_spath = f"{run_dir}/reader_lastmodel.pt"

    print("Saving used nn-arch...")
    with open(f"{run_dir}/used_reader_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(reader.model.__str__())
    with open(f"{run_dir}/used_retriever_arch.txt", 'w', encoding='utf-8') as fd:
        fd.write(retriever.model.__str__())

    print("Saving grad info...")
    reader_grad_info, retriever_grad_info = "", ""
    for name, p in  reader.model.named_parameters():
        reader_grad_info += f"{name} {p.requires_grad}\n"
    with open(f"{run_dir}/reader_gradinfo.txt", 'w', encoding='utf-8') as fd:
        fd.write(reader_grad_info)
    for name, p in  retriever.model.named_parameters():
        retriever_grad_info += f"{name} {p.requires_grad}\n"
    with open(f"{run_dir}/retriever_gradinfo.txt", 'w', encoding='utf-8') as fd:
        fd.write(retriever_grad_info)

    print("Saving used config...")
    with open(f"{run_dir}/used_config.json", 'w', encoding='utf-8') as fd:
        json.dump(config.__dict__, indent=2, fp=fd)

    return logs_file_path, retbmodel_spath, retlmodel_spath, readbmodel_spath, readlmodel_spath 

#
def param_count(model: object) -> int:
    all = sum([p.numel() for name, p in model.named_parameters()])
    trainable = sum([p.numel() for name, p in model.named_parameters() if p.requires_grad])

    return {"all": all, "trainable": trainable}

#
def join_run(config: RunConfig, reader: Union[FiDReader], retriever: Union[BM25E5Retriever, BM25ColBertRetriever], 
             train_loader: DataLoader, eval_loader: DataLoader, train_func: object, evaluate_func: object, 
             reader_metrics: ReaderMetrics, retriever_metrics: RetrievalMetrics) -> Tuple[List[float], List[float], float, List[Dict[str, float]]]:

    logs_fpath, retbm_spath, retlm_spath, readbm_spath, readlm_spath = prepare_join_environment(config, reader, retriever)

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
        if best_score <= eval_scores[-1]['reader'][config.base_score_compare]:
            print("Update Best Model")
            if config.to_save:
                retriever.save_model(retbm_spath)
                reader.save_model(readbm_spath)
            best_score = eval_scores[-1]['reader'][config.base_score_compare]

        # Save train/eval info to logs folder
        save_log(
            logs_fpath, i+1, ml_train[-1], ml_eval[-1], eval_scores[-1], 
            round(train_e - train_s, 5), round(eval_e - train_e, 5))

    print("===LEARNING END===")
    print("Best score: ", best_score)

    print("Save Last Model")
    if config.to_save:
        retriever.save_model(retlm_spath)
        reader.save_model(readlm_spath)

    return ml_train, ml_eval, best_score, eval_scores

#
def join_evaluate(config: RunConfig, reader: Union[FiDReader], retriever: Union[BM25E5Retriever, BM25ColBertRetriever, E5Retriever],  loader: DataLoader,  
                  criterion: object, reader_metrics: ReaderMetrics, retriever_metrics: RetrievalMetrics, epoch: int) -> Tuple[List[float], Dict[str,Dict[str, float]]]:
    reader.model.eval()
    if  not config.retriever_frozen:
        retriever.model.eval()
    
    #
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
    pred_answers = {}
    step = 0

    #
    process = tqdm(loader)
    for batch in process:
        step += 1
        gc.collect()
        torch.cuda.empty_cache()

        # RETRIEVER_PART

        q_ids = batch['q_ids'].to(config.device, non_blocking=True)
        q_masks = batch['q_mask'].to(config.device, non_blocking=True)
        batch['label'] = batch['label'].to(config.device)

        prep_txts = []
        doc_bsz = config.batch_size
        cands_k = config.retrieved_cands
        cands_scores = torch.tensor([], requires_grad=True, device=config.device)
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
            prep_txts += list(map(
                lambda t: config.reader_input_format.format(q=batch['q_text'][i],c=t), texts))
        
        # READER_PART

        tokenized_txts = reader.tokenize(prep_txts)
        tokenized_txts = {k: v.to(config.device) for k, v in tokenized_txts.items()}

        if config.retriever_frozen:

            # 
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
                labels=batch['label'], # doc_bsz x seq_len
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
            #print("reader topk_shape: ", output.logits.shape) # doc_bsz x seq_len x vocab_size 
            
            #
            loss = output.loss
        else:

            #
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz*cands_k, 1, -1), 
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz*cands_k, 1, -1),
                labels=batch['label'].repeat(1, cands_k).view(doc_bsz*cands_k,-1))
            #print("reader k_shape: ", output.logits.shape) # doc_bsz * cands_k x seq_len x vocab_size 

            #
            seq_len = output.logits.shape[1]
            reader_k_loss = criterion.k_loss(
                output.logits.view(doc_bsz, cands_k, seq_len,-1), 
                batch['label']).detach()

            #print("ids: ", tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1).shape)
            #print("mask: ", tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1).shape)
            #print("label: ", batch['label'].shape)

            #
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
                labels=batch['label'], 
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
            #print("reader topk_shape: ", output.logits.shape) # doc_bsz x seq_len x vocab_size 

            reader_topk_loss = output.loss
            
            # Compute Join-loss
            loss = criterion(reader_topk_loss, reader_k_loss, cands_scores.view(doc_bsz, cands_k))

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        # Generating Answers by predicted indices
        output = reader.model.generate(
            input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1),
            attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1), 
            max_length=config.reader_gen_ml, eos_token_id=reader.tokenizer.eos_token_id)
        predicted = reader.tokenizer.batch_decode(output, skip_special_tokens=True)

        # Calculating metrics 
        pred_answers[step] = {'gen':predicted, 'target': batch['label_text']}
        scores['reader']['bleu'].append(reader_metrics.bleu(predicted,batch['label_text']))
        scores['reader']['rouge'].append(reader_metrics.rouge(predicted,batch['label_text']))
        scores['reader']['meteor'].append(reader_metrics.meteor(predicted,batch['label_text']))
        scores['reader']['em'].append(reader_metrics.exact_match(predicted,batch['label_text']))

    print("Saving generated answers during evaluation...")
    with open(f"{config.base_dir}/logs/{config.run_name}/gen_answers_epoch{epoch}.json", 
              'w', encoding='utf-8') as fd:
        json.dump(pred_answers, indent=2, fp=fd)

    # Averaging computed metrics
    scores['retriever']['MRR'] = round(np.mean(scores['retriever']['MRR']), 5)
    scores['retriever']['mAP'] = round(np.mean(scores['retriever']['MRR']), 5)
    scores['reader']['bleu'] = round(np.mean(scores['reader']['bleu']), 5)
    scores['reader']['rouge'] = round(np.mean(scores['reader']['rouge']), 5)
    scores['reader']['meteor'] = round(np.mean(scores['reader']['meteor']), 5)
    scores['reader']['em'] =  round(np.mean(scores['reader']['em']), 5)

    return losses, scores

#
def join_train(config: RunConfig, reader: Union[FiDReader], 
               retriever: Union[BM25E5Retriever, BM25ColBertRetriever, E5Retriever], 
               loader: DataLoader, optimizer: object, criterion: JoinLoss) -> List[float]:
    reader.model.train()
    if  not config.retriever_frozen:
        retriever.model.train()
    
    #
    losses = []
    optimizer.zero_grad()

    #
    process = tqdm(loader)
    for step, batch in enumerate(process):
        gc.collect()
        torch.cuda.empty_cache()

        # RETRIEVER_PART

        q_ids = batch['q_ids'].to(config.device, non_blocking=True)
        q_masks = batch['q_mask'].to(config.device, non_blocking=True)
        batch['label'] = batch['label'].to(config.device)

        prep_txts = []
        doc_bsz = config.batch_size
        cands_k = config.retrieved_cands
        cands_scores = torch.tensor([], requires_grad=True, device=config.device)
        for i in range(len(batch['q_text'])):
            texts, k_scores, _ = retriever.search(
                batch['q_text'][i], {'input_ids': q_ids[i].unsqueeze(0), 
                                     'attention_mask':q_masks[i].unsqueeze(0)})

            cands_scores = torch.cat((cands_scores, k_scores), dim=0)
            prep_txts += list(map(
                lambda t: config.reader_input_format.format(q=batch['q_text'][i],c=t), texts))
        
        # READER_PART

        tokenized_txts = reader.tokenize(prep_txts)
        tokenized_txts = {k: v.to(config.device) for k, v in tokenized_txts.items()}

        #print("tokenized_txts ids: ", tokenized_txts['input_ids'].shape)

        if config.retriever_frozen:

            #
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
                labels=batch['label'], 
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
            #print("reader topk_shape: ", output.logits.shape) # doc_bsz x seq_len x vocab_size 

            #
            loss = output.loss

        else: 

            #
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz*cands_k, 1, -1), 
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz*cands_k, 1, -1),
                labels=batch['label'].repeat(1, cands_k).view(doc_bsz*cands_k,-1))
            #print("reader k_states: ", output.logits.shape) # doc_bsz * k_cands x seq_len x vocab_size

            #
            seq_len = output.logits.shape[1]
            reader_k_loss = criterion.k_loss(
                output.logits.view(doc_bsz, cands_k, seq_len,-1),
                batch['label']).detach()
            
            #print("ids: ", tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1).shape)
            #print("mask: ", tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1).shape)
            #print("label: ", batch['label'].shape)

            #
            output = reader.model(
                input_ids=tokenized_txts['input_ids'].view(doc_bsz, cands_k, -1), 
                labels=batch['label'], 
                attention_mask=tokenized_txts['attention_mask'].view(doc_bsz, cands_k, -1))
            #print("reader topk_shape: ", output.logits.shape) # doc_bsz x seq_len x vocab_size 

            #
            reader_topk_loss = output.loss

            # Compute Join-loss
            loss = criterion(reader_topk_loss, reader_k_loss, cands_scores.view(doc_bsz, cands_k))
            #print("Computed join-loss: ", loss)

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})

        loss = loss / config.grad_accum_steps
        loss.backward()

        if ((step+1)%config.grad_accum_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

    return losses

