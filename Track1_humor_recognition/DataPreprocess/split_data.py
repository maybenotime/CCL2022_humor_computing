import torch
import json
from torch.utils.data import random_split

train_path = '../CoG-BART/data/humor/train_data.json'
valid_path = '../CoG-BART/data/humor/dev_data.json'
test_path = '../CoG-BART/data/humor/test_data.json'

def load_data(path):
    with open(path,"r") as f:
        data = json.load(f)       
    return data
    

def split(data):
    train_size = int(len(data)*0.8)
    valid_size = int(len(data)*0.1)
    test_size = len(data) - train_size - valid_size
    train_dataset, valid_data, test_dataset = random_split(
    dataset=data,
    lengths=[train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(0)
    )
    train_dataset = list(train_dataset)
    valid_data = list(valid_data)
    test_dataset = list(test_dataset)
    with open(train_path,"w") as w_train, open(valid_path,"w") as w_valid, open(test_path,"w") as w_test:
        json.dump(train_dataset, w_train)
        json.dump(valid_data, w_valid)
        json.dump(test_dataset, w_test)
            
data = load_data('../use_data/friends.json')
split(data)

