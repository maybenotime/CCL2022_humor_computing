import os 

save_path = './clean_data/train_dev.txt'

def main():
    all_data = []
    data_dir = './clean_data'
    txt_list = os.listdir(data_dir)
    for path in txt_list:
        file_path = os.path.join(data_dir,path)
        with open(file_path,"r") as f:
            for line in f:
                line = line.strip()
                all_data.append(line)
    
    with open(save_path,"w") as w:
        for line in all_data:
            w.write(line)
            w.write('\n')
main()