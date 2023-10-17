from torchtext import data
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext
import torch
import json

def strtoint(label):
    return [int(float(label[0]))]

def get_dataset(json_data,utter_field,label_field):
    field = {'Sentence':('sen',utter_field),'Speaker':None,'Label':('label',label_field)}
    examples = []
    with open(json_data,"r") as f:
        dialogue_list = json.load(f)
    for dialogue in dialogue_list:
        for utter in dialogue:
            utter = json.dumps(utter)
            examples.append(data.Example.fromJSON(utter,field))
    new_field = {'sen':utter_field,'label':label_field}
    return examples,new_field

def build_vocab(train,valid,vectors,specials=None):
    if specials is None:
        specials = ['<unk>', '<pad>']
    counter = Counter()
    for example in train:
        word_list = vars(example)['sen']
        counter.update(word_list)
    for example in valid:
        word_list = vars(example)['sen']
        counter.update(word_list)
    vocabs = Vocab(counter,specials=specials,vectors=vectors)
    return vocabs

def get_iter(config,args):
    stopwords = open(config.stopwords_path).read().split('\n')
    tokenizer = get_tokenizer('basic_english')
    UTTER = data.Field(sequential=True,lower=True,tokenize=tokenizer,stop_words=stopwords)    
    LABEL = data.Field(preprocessing=strtoint,use_vocab=False)
    train_examples, train_fields = get_dataset(config.train_name, UTTER, LABEL)
    valid_examples, valid_fields = get_dataset(config.dev_name, UTTER, LABEL)
    print(vars(train_examples[0]))
    train_dataset = data.Dataset(train_examples, train_fields)
    valid_dataset = data.Dataset(valid_examples, valid_fields)

    vectors = torchtext.vocab.Vectors(name=config.pretrained_name,cache=config.pretrained_path)
    UTTER.build_vocab(train_dataset, valid_dataset,vectors=vectors)
    LABEL.build_vocab(train_dataset, valid_dataset)

    train_iter, dev_iter = \
	data.BucketIterator.splits(
        (train_dataset, valid_dataset),   #需要生成迭代器的数据集
        batch_sizes=(args.batch_size,args.batch_size),                  # 每个迭代器分别以多少样本为一个batch,验证集和测试集数据不需要训练，全部放在一个batch里面就行了
        sort_key=lambda x: len(x.sen),           #按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
        sort_within_batch = True
        )

    return UTTER, LABEL,train_iter,dev_iter