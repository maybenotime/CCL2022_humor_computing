import os
import json 
 
data_dir = '../train_dev'
save_path = '../train_dev/train_dev_data.json'

def load_dir_data():
    file_list = os.listdir(data_dir)
    all_data = []
    for file_name in file_list:
        path = os.path.join(data_dir,file_name)
        with open(path,"r") as f:
            file_data = json.load(f)
        print(len(file_data))
        all_data.extend(file_data)
    print(len(all_data))
    return all_data    
if __name__ == '__main__':
    data = load_dir_data()
    with open(save_path,"w") as w:
        json.dump(data,w)