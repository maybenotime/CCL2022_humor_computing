from fastNLP import cache_results
from utils.pipe import BertPipe 
from fastNLP import Instance, DataSet
from predictor.legal_predictor import Predictor 
from transformers import BertTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score 
import torch.nn.functional as F
import torch
import csv
import json
import numpy as np
inference_result = './result/Tal2022_任务1_prob.csv'
data_path = './data/ensembel_dev_data.json'
id_path = './data/id.json'

######
lr = 4e-6
n_epochs = 10
bert_model = 'bert-large-uncased'
batch_size = 8
up_num =0
######
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_dialogue_modeling/save_models/best_dialogue_modeling_mixscore_2022-09-15-10-12-54-796442'
model = torch.load(checkpoint)
tokenizer = BertTokenizer.from_pretrained(bert_model)

sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
unk_id = tokenizer.convert_tokens_to_ids(['[UNK]'])[0] 

model.to(device=device)
print("model loaded")

with open(data_path, 'r') as f:
    data = json.load(f)
    dialogue_acc_list = []          #计算dialogue级的acc
    all_preds = []
    all_label = []
    all_probs = []                  #概率
    all_ids = []
    for dialogue in data:
        dataset = DataSet()
        for utter in dialogue:
            # id = utter['id']
            sen = utter["Sentence"]
            speaker = utter["Speaker"]
            sen_speaker = speaker + ':' + sen
            sen_with_speaker = tokenizer.tokenize(sen_speaker)
            sen_with_speaker = tokenizer.convert_tokens_to_ids(sen_with_speaker)
            input = [cls_id] + sen_with_speaker + [sep_id]  #构造input
            input = input[:256]                                    #截断，最大输入长度是256
            label = int(float(utter["Label"]))
            ins = Instance(input = input, label = label)
            dataset.append(ins)

        dataset.set_pad_val('input', pad_id)    #对于input这个field,使用pad_id来做填充
        dataset.set_input('input')              #将该field设置为input, 对data_bundle中所有的dataset执行
        dataset.set_target('label')
        dataset.add_seq_len('input')
        infer = Predictor(
            network=model,
            batchsize=4
        )
        result = infer.predict(dataset)
        target_label = []
        for ins in dataset:
            target = ins['label']
            target_label.append(target)
        ids = []
        # for ins in dataset:
        #     id = ins['id']
        #     ids.append(id)
        dialogue_label = []
        dialogue_prob = []
        for batch_array in result['pred']:
            for array in batch_array:
                label = np.argmax(array)
                prob_array = F.softmax(torch.tensor(array))
                prob = float(prob_array[1])
                dialogue_label.append(label)
                dialogue_prob.append(prob)
        acc = sum([int(i==j) for i,j in zip(target_label, dialogue_label)])/len(dialogue_label)
        # if acc < 0.8:
        #     if len(dataset) % 4 == 1:
        #         print(len(dataset))
        #         # print(dialogue_label)
        #         # print(target_label)
        all_preds.extend(dialogue_label)
        all_probs.extend(dialogue_prob)
        all_label.extend(target_label)
        # all_ids.extend(ids)
        dialogue_acc_list.append(acc)
    final_acc = sum(dialogue_acc_list)/len(dialogue_acc_list)
    
    acc = accuracy_score(all_label, all_preds)
    f1 = f1_score(all_label, all_preds, average='binary')
    recall = recall_score(all_label, all_preds, average='binary')
    pre = precision_score(all_label, all_preds, average='binary')
    print("final_acc:{},f1:{},acc:{},recall:{},pre:{}".format(final_acc,f1,acc,recall,pre))

    csv_data = []
    for label,prob in zip(all_preds,all_probs):
        row = [label,prob]
        csv_data.append(row)

    header = ['Label','Prob']
    with open(inference_result,"w",encoding='utf-8',newline='') as w:
        writer = csv.writer(w)
        writer.writerow(header)
        writer.writerows(csv_data)
    
    # print(all_ids)
    # with open(id_path,"w") as w:
    #     json.dump(all_ids,w)


