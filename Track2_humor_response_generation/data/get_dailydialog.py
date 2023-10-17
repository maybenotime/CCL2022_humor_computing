from datasets import load_dataset
import json

save_path = './dailydialog.json'

dataset = load_dataset("daily_dialog")   #加载数据集
train_data = dataset['train']['dialog']

with open(save_path,"w") as w:
    json.dump(train_data,w)