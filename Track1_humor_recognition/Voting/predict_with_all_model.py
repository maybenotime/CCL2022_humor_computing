import csv 
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#投票，加权融合logit(两种方式) 都尝试了

bert_utter_result = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_utter/result/Tal2022_任务1_prob.csv'
bert_one_sen_result = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_one_senten/result/Tal2022_任务1_prob.csv'
bert_dialog_result = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_dialogue_modeling/result/Tal2022_任务1_prob.csv'
cog_bart_result = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/CoG_BART_humor/test_result/task1_prob.csv'

id_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/bert_humor_dialogue_modeling/data/id.json'

save_path = './Tal2022_任务1.csv'

def load_predict_result(result_path):
    with open(result_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        all_prob = []
        all_pred = []
        header = next(csv_reader)        # 读取第一行每一列的标题 
        for row in csv_reader:
            prob = round(float(row[1]),4)
            all_prob.append(prob)
            pred = row[0]
            all_pred.append(pred)
    return all_prob,all_pred

def load_predict_result_plus_gt(result_path):
    with open(result_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        all_prob = []
        all_pred = []
        gts= []
        header = next(csv_reader)        # 读取第一行每一列的标题 
        for row in csv_reader:
            prob = round(float(row[1]),4)
            all_prob.append(prob)
            pred = row[0]
            all_pred.append(pred)
            gt = int(row[2])
            gts.append(gt)
    return all_prob,all_pred,gts

top1_prob,top1_pred = load_predict_result(bert_utter_result)
top2_prob,top2_pred,gts = load_predict_result_plus_gt(bert_one_sen_result)
top3_prob,top3_pred = load_predict_result(bert_dialog_result)
top4_prob,top4_pred = load_predict_result(cog_bart_result)

final_test_result = []
# for t1,t2,t3,t4 in zip(top1_prob,top2_prob,top3_prob,top4_prob):
#     final_prob = 0.28*t1 + 0.26*t2 + 0.24*t3 + 0.22*t4
#     final_prob = round(final_prob,4)
#     if final_prob > 0.5:
#         label = 1
#     else:
#         label = 0
#     final_test_result.append(label)
up = 0
down = 0
for index,tuple in enumerate(zip(top1_pred,top2_pred,top3_pred,top4_pred,gts)):
    tuple = list(map(int,tuple))
    # print(tuple)
    vote = sum(tuple[:-1])
    gt = tuple[-1]
    if vote >= 3:
        label = 1
    elif vote <= 1:
        label = 0
    else:
        label = tuple[0]
    if tuple[0] == gt and label != gt:
        # print(tuple,index)
        down += 1
    if tuple[0] != gt and label == gt:
        up += 1 
    final_test_result.append(label)

acc = accuracy_score(gts, final_test_result)
f1 = f1_score(gts, final_test_result, average='binary')
precision = precision_score(gts,final_test_result, average='binary')
recall = recall_score(gts,final_test_result, average='binary')
print(up,down)
print('acc:{}f1:{}precision:{}recall:{}'.format(acc,f1,precision,recall))
# with open(id_path,"r") as f:
#     all_ids = json.load(f)


# csv_data = []
# for id,label in zip(all_ids,final_test_result):
#     row = [id,label]
#     csv_data.append(row)

# header = ['ID','Label']
# with open(save_path,"w",encoding='utf-8',newline='') as w:
#     writer = csv.writer(w)
#     writer.writerow(header)
#     writer.writerows(csv_data)



