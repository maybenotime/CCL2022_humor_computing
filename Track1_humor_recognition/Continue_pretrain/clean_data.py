import re
import os

save_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/pretrain_data/clean_data/big_boom_4_12.txt'

def dialogue_fileter(path):
    data = []
    with open(path,"r") as f:
        for line in f:
            line = line.strip()
            if line[:8] == 'Dialogue':
                data.append(line)
    return data

def en_filter(data):
    dialogue_data = []
    for line in data:
        pattern = r'c&H4bc2ef&}(.+)$' #对话前都是这个符号
        groups = re.search(pattern,line)
        if groups:
            dialogue = groups.group(1)
            print(dialogue)
            dialogue_data.append(dialogue)
    return dialogue_data
    
def main():
    data_dir = "./captions"
    file_list = os.listdir(data_dir)
    all_data = []
    for path in file_list:
        file_path = os.path.join(data_dir,path)
        data =dialogue_fileter(file_path)
        dialogue_data = en_filter(data)
        all_data.append(dialogue_data)
    with open(save_path,"w") as w:
        for part_data in all_data:
            for line in part_data:
                w.write(line)
                w.write('\n')

main()