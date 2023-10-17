from torch.utils.data import Dataset
from tqdm import tqdm

class DialogData(Dataset):
    def __init__(self,dataset,humor_json,tokenizer):
        super().__init__()
        self.model_inputs = []
        if dataset is not None:
            for utter_list in tqdm(dataset):
                if len(utter_list) < 4:     #舍弃上文不足三句的dialog
                    continue
                j = 3                   #j是滑动窗口指针
                j_limitation = len(utter_list) - 1
                while j <= j_limitation:
                    example = {}
                    context = utter_list[j-3:j]
                    context_input_ids = []
                    resp = 'Speaker_B' + ':' + utter_list[j]
                    respone_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(resp))
                    respone_ids.append(1)       
                    for idx,sen in enumerate(context):
                        sen += '</s>'           #分割utter
                        if idx % 2 == 0:
                            speaker = 'Speaker_A'
                        else:
                            speaker = 'Speaker_B'
                        sen_with_speaker = speaker + ':' + sen
                        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen_with_speaker))
                        context_input_ids.extend(input_ids)
                    example['input_ids'] = context_input_ids
                    example['labels'] = respone_ids
                    self.model_inputs.append(example)
                    j += 1

        for utter_list in tqdm(humor_json):
            if len(utter_list) < 4:     #舍弃上文不足三句的dialog
                continue
            j = 3                   #j是滑动窗口指针
            j_limitation = len(utter_list) - 1
            while j <= j_limitation:
                example = {}
                context = utter_list[j-3:j]
                context_str = []
                for utter in context:
                    sen_with_speaker = utter['Speaker'] + ':' + utter['Sentence']
                    context_str.append(sen_with_speaker)
                context_input_ids = []
                resp = utter_list[j]['Speaker'] + ':' + utter_list[j]['Sentence']
                respone_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(resp))
                respone_ids.append(1)       
                for sen in context_str:
                    sen += '</s>'           #分割utter
                    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))
                    context_input_ids.extend(input_ids)
                example['input_ids'] = context_input_ids
                example['labels'] = respone_ids
                self.model_inputs.append(example)
                j += 1

        print(len(self.model_inputs))
                

    
    def __len__(self):
        return len(self.model_inputs)
    
    def __getitem__(self,index):
        return self.model_inputs[index]
        
            
