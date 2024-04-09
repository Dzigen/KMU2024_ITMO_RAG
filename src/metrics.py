# Source: https://amitness.com/2020/08/information-retrieval-evaluation/

#Retrieval metrics
# - mAP
# - MRR
# - precision
#Reader metrics
# - BLEU presision
# - ROUGE recall
# - METEOR f1

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
import evaluate
import numpy as np

#
class RetrievalMetrics:
    def __init__(self):
        pass
    
    def precision(self, predicted_cands, gold_cands, k):
        true_positive = np.isin(predicted_cands[:k], gold_cands).sum()
        false_positive = k - true_positive
        return round(true_positive / (true_positive + false_positive),5)

    def AP(self, predicted_cands, gold_cands):
        indicators = np.isin(predicted_cands, gold_cands)

        numerator = np.sum([self.precision(predicted_cands, gold_cands, k+1) 
                            for k in range(len(predicted_cands)) if indicators[k]])

        return round(numerator / len(gold_cands), 5)

    def mAP(self, predicted_cands, gold_cands):
        return np.mean(self.AP(predicted_cands, gold_cands))

    def reciprocal_rank(self, predicted_cands, gold_cands):
        indicators = np.isin(predicted_cands, gold_cands)
        first_occur = np.where(indicators == True)[0]
        return round(0 if first_occur.size == 0 else 1 / (first_occur[0] + 1), 5)
        
    def MRR(self, predicted_cands, gold_cands):
        return np.mean(self.reciprocal_ranks(predicted_cands, gold_cands))

#
class ReaderMetrics:
    def __init__(self):
        self.rouge_obj = ROUGEScore()
        self.bleu_obj = BLEUScore(n_gram=2)
        self.meteor_obj = evaluate.load('meteor')
        self.em_obj = evaluate.load("exact_match")

    def rouge(self, predicted, targets):
        accum = []
        for i in range(len(targets)):
            accum.append(self.rouge_obj(predicted[i], targets[i])['rougeL_fmeasure'])
        return round(np.mean(accum),5)

    def bleu(self, predicted, targets):
        accum = []
        for i in range(len(targets)):
            accum.append(self.bleu_obj([predicted[i]], [[targets[i]]]))
        return round(np.mean(accum),5)

    def meteor(self, predicted, targets):
        accum = []
        for i in range(len(targets)):
            accum.append(
                self.meteor_obj.compute(
                    predictions=[predicted[i]], references=[targets[i]])['meteor'])
        return round(np.mean(accum),5)
    
    def exact_match(self, predicted, targets):
        return round(self.em_obj(predictions=predicted, references=targets)["exact_match"], 2)