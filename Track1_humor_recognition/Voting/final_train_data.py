import csv
import json
import re
import os

train_dir='/aliyun-06/share_v2/yangshiping/projects/humor_computation/final_dataset/dataset/task2_test'
save_path = '../data/train_2.json'



def organize_data(filepath):
    file_data = []
    speaker_set = set()                     #说话人集合
    id_set = set()
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题 
        col_name = ['Sentence', 'Speaker', 'Label']    #我们需要的列
        header_dic = {}
        for index,col in enumerate(header):
            header_dic[col] = index
        for row in csv_reader:
            try:
                d_id = int(float(row[header_dic['Dialogue_id']]))
            except:
                print(filepath)
            utter = {}     
            speaker = row[header_dic['Speaker']]
            speaker_set.add(speaker)
            for col in col_name:
                utter[col] = row[header_dic[col]]
            if d_id not in id_set:
                id_set.add(d_id)
                if len(id_set) == 1:
                    dialogue = []
                    dialogue.append(utter)
                else:
                    file_data.append(dialogue)
                    dialogue = []
                    dialogue.append(utter)
            else:
                dialogue.append(utter)
        file_data.append(dialogue)       #加上最后一个dialogue 

        return file_data, speaker_set

def main():
    file_list = os.listdir(train_dir)
    sorted_list = sorted(file_list)
    all_data = []   
    train_speaker_set = set()
    for file in sorted_list:
        file_path = os.path.join(train_dir, file)
        file_data, speaker_set = organize_data(file_path)
        all_data.append(file_data)
        train_speaker_set = train_speaker_set.union(speaker_set)
    
    total_dialogue = []
    for dia_list in all_data:           #把列表的列表转成列表
        for dia in dia_list:
            total_dialogue.append(dia)
    
    with open(save_path,"w",encoding='utf-8') as w:
        json.dump(total_dialogue,w)

main()