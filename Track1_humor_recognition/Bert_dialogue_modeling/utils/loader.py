import os
import json
from fastNLP import Instance, DataSet
from fastNLP.io import Loader, DataBundle
from collections import deque
import copy

class dataLoader(Loader):
    def __init__(self):
        super().__init__()

    def load(self, folder, up_num):
        data_bundle = DataBundle()
        file_list = ['train.json','valid.json','test.json']
        for file in file_list:
            path = os.path.join(folder, file)
            with open(path, 'r') as f:
                data = json.load(f)
                dataset = DataSet()
                count_pass = 0      #跳过了多少对话
                for dialogue in data:
                    queue = deque(maxlen=4)         #数据增强的窗口为4
                    if len(dialogue) < 4:           #跳过长度小于4的对话
                        print("跳过长度小于4的对话！")
                        count_pass += 1
                        continue
                    for utter in dialogue:
                        sen = utter["Sentence"]
                        speaker = utter["Speaker"]
                        sen_speaker = speaker + ':' + sen
                        label = int(float(utter["Label"]))
                        ins = Instance(sentence = sen_speaker, label = label)
                        queue.append(ins)
                        if len(queue) == 4:
                            dead_queue = copy.deepcopy(queue)
                            for ins in dead_queue:
                                 dataset.append(ins)
            data_bundle.set_dataset(dataset, name=file.split('.')[0])
    # 读取全部说话人
        speakers = []
        with open(os.path.join(folder, 'train_speaker.json'), 'r') as f:
            speaker_dic = json.load(f)
            for speaker in speaker_dic.keys():
                speakers.append(speaker)
        setattr(data_bundle, 'speakers', speakers) #为data_bundle对象设置一个属性'speakers'
        return data_bundle