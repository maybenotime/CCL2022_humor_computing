from curses.ascii import isdigit
import os 
import re 

save_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/pretrain_data/clean_data/growing_pains.txt'

def dialogue_fileter(path):
    data = []
    with open(path,"r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if isdigit(line[0]):
                continue
            if line[0] == '/':
                continue
            data.append(line)
    return data


def main():
    data_dir = "./captions"
    file_list = os.listdir(data_dir)
    all_data = []
    for path in file_list:
        file_path = os.path.join(data_dir,path)
        print(file_path)
        data =dialogue_fileter(file_path)
        all_data.append(data)
    with open(save_path,"w") as w:
        for part_data in all_data:
            for line in part_data:
                w.write(line)
                w.write('\n')

main()