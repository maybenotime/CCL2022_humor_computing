from collections import deque
import csv
import json
import re
import os
from urllib import response

train_dir='/aliyun-06/share_v2/yangshiping/projects/humor_computation/final_dataset/dataset/task2_test'
save_path = '../data/task2_test.json'


def organize_data(filepath):
    response_data = []
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题 
        print(header)
        col_name = ['Dialogue_id','Sentence', 'Speaker']    #我们需要的列
        header_dic = {}
        for index,col in enumerate(header):
            header_dic[col] = index
        context_queue = deque(maxlen=3)         #上下文的窗口为3
        for row in csv_reader:
            utter = {}     
            for col in col_name:
                utter[col] = row[header_dic[col]]
            if utter['Sentence'] == 'NULL':
                utter['context'] = list(context_queue)
                print(utter)
                response_data.append(utter)
            else:
                text = utter['Speaker'] + ':' + utter['Sentence']
                context_queue.append(text)
        return response_data

def main():
    file_list = os.listdir(train_dir)
    sorted_list = sorted(file_list)
    all_data = []   
    for file in sorted_list:
        file_path = os.path.join(train_dir, file)
        response_data = organize_data(file_path)
        all_data.append(response_data)

    
    total_dialogue = []
    for dia_list in all_data:           #把列表的列表转成列表
        for dia in dia_list:
            total_dialogue.append(dia)

    with open(save_path,"w",encoding='utf-8') as w:
        json.dump(total_dialogue,w)


main()