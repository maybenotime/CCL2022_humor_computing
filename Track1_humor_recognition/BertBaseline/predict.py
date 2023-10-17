from fastNLP import cache_results
from utils.pipe import BertPipe 
from predictor.legal_predictor import Predictor 
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score 
import torch
import torch.nn.functional as F
import csv
import numpy as np
inference_result = './result/Tal2022_任务1_prob.csv'
data_path = './data'

######
lr = 2e-5
n_epochs = 20
bert_model = 'bert-large-uncased'
batch_size = 16
up_num =3
######
@cache_results('caches/up_{}.pkl'.format(up_num), _refresh=True)        #将预处理的数据缓存
def get_data():
    data_bundle = BertPipe(bert_model, up_num).process_from_file(data_path)
    return data_bundle

data_bundle = get_data()
dev_data = data_bundle.get_dataset('test')
dialog_len_list = data_bundle.dev_dialogue_len      #验证集每个dialogue的长度
dev_data.add_seq_len('input')
sum1 = 0
dev_split = []
for num in dialog_len_list: 
    sum1 += num
    dev_split.append(sum1)
start = 0
splited_dev = []            #其中存储着每个dialogue的列表
for pos in dev_split:
    part = dev_data[start:pos]
    start = pos
    splited_dev.append(part)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_one_senten/save_models_2/best_bertcls_mixscore_2022-08-26-12-39-54-138551'
model = torch.load(checkpoint)
model.to(device=device)
print("model loaded")

infer = Predictor(
    network=model,
)

all_preds = []
all_label = []
all_probs = []
dialogue_acc_list = []
for dialogue in splited_dev:
    result = infer.predict(dialogue)
    target_label = []
    for ins in dialogue:
        target = ins['label']
        target_label.append(target)
    pred_label = []
    probs = []
    for batch_array in result['pred']:
        for array in batch_array:
            label = np.argmax(array)
            prob_array = F.softmax(torch.tensor(array))
            prob = float(prob_array[1])
            probs.append(prob)
            pred_label.append(label)
                
    acc = sum([i==j for i,j in zip(target_label, pred_label)])/len(pred_label)
    all_preds.extend(pred_label)
    all_probs.extend(probs)
    all_label.extend(target_label)
    dialogue_acc_list.append(acc)

final_acc = sum(dialogue_acc_list)/len(dialogue_acc_list)
acc = accuracy_score(all_label, all_preds)
f1 = f1_score(all_label, all_preds, average='binary')
recall = recall_score(all_label, all_preds, average='binary')
pre = precision_score(all_label, all_preds, average='binary')
print("final_acc:{},f1:{},acc:{},recall:{},pre:{}".format(final_acc,f1,acc,recall,pre))
print("mixscore:{}".format(final_acc+f1))


csv_data = []
for label,prob,gt in zip(all_preds,all_probs,all_label):
    row = [label,prob,gt]
    csv_data.append(row)

header = ['Label','Prob','GT']
with open(inference_result,"w",encoding='utf-8',newline='') as w:
    writer = csv.writer(w)
    writer.writerow(header)
    writer.writerows(csv_data)

