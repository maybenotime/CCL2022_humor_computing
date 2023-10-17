from fastNLP import MetricBase
import torch
import wandb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score 

class MixMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self.preds = []
        self.labels = []
        self.best_score = 0
        self.best_f1 = 0

    def evaluate(self, pred, label):
        pred_list = torch.argmax(pred,dim=-1).cpu().tolist()
        self.preds.extend(pred_list)
        label_list = label.tolist()
        self.labels.extend(label_list)

    def get_metric(self, reset=True):
        acc = accuracy_score(self.labels, self.preds)
        f1 = f1_score(self.labels, self.preds, average='binary')
        recall = recall_score(self.labels, self.preds, average='binary')
        pre = precision_score(self.labels, self.preds, average='binary')
        mixscore = acc + f1
        if mixscore > self.best_score:
            self.best_score = mixscore
            wandb.run.summary["best_mixscore"] = self.best_score
        if f1 > self.best_f1:
            self.best_f1 = f1
            wandb.run.summary["best_f1"] = self.best_f1
        if reset:
            self.preds= []
            self.labels = []
        
        result = {"mixscore":mixscore, "f1":f1, "acc":acc, "recall":recall, "precision":pre}
        wandb.log(result)
        return result