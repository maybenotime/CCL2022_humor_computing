import torch
from torch.utils.data import random_split

train_path = './clean_data/train.txt'
valid_path = './clean_data/dev.txt'

def load_data(path):
    data = []
    with open(path,"r") as f:
        for line in f:
            line = line.strip()
            data.append(line)
    return data
    

def split(data):
    train_size = int(len(data)*0.95)
    valid_size = len(data) - train_size
    train_dataset, valid_data = random_split(
    dataset=data,
    lengths=[train_size, valid_size],
    generator=torch.Generator().manual_seed(0)
    )
    with open(train_path,"w") as w_train, open(valid_path,"w") as w_valid:
        for sen in train_dataset:
            w_train.write(sen)
            w_train.write('\n')
        
        for sen in valid_data:
            w_valid.write(sen)
            w_valid.write('\n')

            
data = load_data('./clean_data/train_dev.txt')
split(data)

