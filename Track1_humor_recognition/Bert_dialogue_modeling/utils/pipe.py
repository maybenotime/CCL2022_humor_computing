from .loader import dataLoader
from fastNLP import DataSet
from fastNLP.io import  DataBundle, Pipe
from transformers import BertTokenizer, BertModel

    
    
def _prepare_data_bundle(tokenizer, data_bundle):
    sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    unk_id = tokenizer.convert_tokens_to_ids(['[UNK]'])[0] 

    for name in data_bundle.get_dataset_names():    #重构各个数据集，field主要有input和target
        ds = data_bundle.get_dataset(name)
        new_ds = DataSet()
        for ins in ds:
            sen_with_speaker = ins['sentence']
            sen_with_speaker = tokenizer.tokenize(sen_with_speaker)
            sen_with_speaker = tokenizer.convert_tokens_to_ids(sen_with_speaker)
            input = [cls_id] + sen_with_speaker + [sep_id]  #构造input
            input = input[:256]                                    #截断，最大输入长度是256
            ins['input'] = input
            new_ds.append(ins)
        data_bundle.set_dataset(new_ds, name)



    data_bundle.set_pad_val('input', pad_id)    #对于input这个field,使用pad_id来做填充
    data_bundle.set_input('input')              #将该field设置为input, 对data_bundle中所有的dataset执行
    data_bundle.set_target('label')
    setattr(data_bundle, 'pad_id', pad_id)


    return data_bundle

class BertPipe(Pipe):
    def __init__(self, bert_dir, up_num):
        super().__init__()
        self.bert_dir = bert_dir
        self.up_num = up_num

    def process(self, data_bundle):
        tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        return _prepare_data_bundle(tokenizer, data_bundle)

    def process_from_file(self, path) -> DataBundle:
        data_bundle = dataLoader().load(path,self.up_num)
        return self.process(data_bundle)