from transformers import AutoModel,BertConfig
from torch import nn

class bertcls(nn.Module):
    def __init__(self, bert_model, pad_id):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.pad_id = pad_id
        # self.cls = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 256),
        #                          nn.GELU(),
        #                          nn.Dropout(0.3),
        #                          nn.Linear(256,64),
        #                          nn.GELU(),
        #                          nn.Dropout(0.3),
        #                          nn.Linear(64,2))
        self.cls = nn.Linear(self.bert.config.hidden_size,2)
    def forward(self,input,token_type_ids):
        '''
        补充上文信息时需要再处理一下token_type_ids
        '''
        attention_mask = input.ne(self.pad_id)      #避免对pad id执行attention
        return_dic = self.bert(input_ids=input, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        # cls_pooler_output = return_dic['pooler_output']       #bert部分
        # logits = self.cls(cls_pooler_output)
        cls_pooler_output = return_dic['last_hidden_state']
        logits = self.cls(cls_pooler_output[:,0,:])     #取cls token
        return {"pred": logits}
    
 

