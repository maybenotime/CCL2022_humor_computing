import os
import json
from fastNLP import Instance, DataSet
from fastNLP.io import Loader, DataBundle
from collections import deque

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
                if file == 'test.json':
                    dialogue_len_list = []
                for dialogue in data:
                    if file == 'test.json':
                        dialogue_len_list.append(len(dialogue))
                    context_deque = deque(maxlen=up_num)            #使用一个长度固定的队列存储上文信息
                    for utter in dialogue:
                        sen = utter["Sentence"]
                        speaker = utter["Speaker"]
                        sen_plus = speaker + ':' + sen
                        if len(context_deque)==0:               
                            context_str = "dialogue begin."
                        else:
                            context_str = '|'.join(context_deque)
                        context_deque.append(sen_plus)
                        label = int(float(utter["Label"]))
                        ins = Instance(sentence = sen, speaker = speaker, context=context_str, label = label)
                        dataset.append(ins)
            data_bundle.set_dataset(dataset, name=file.split('.')[0])
    # 读取全部说话人
        speakers = []
        with open(os.path.join(folder, 'train_speaker.json'), 'r') as f:
            speaker_dic = json.load(f)
            for speaker in speaker_dic.keys():
                speakers.append(speaker)
        setattr(data_bundle, 'speakers', speakers) #为data_bundle对象设置一个属性'speakers'
        setattr(data_bundle, 'dev_dialogue_len', dialogue_len_list)
        return data_bundle