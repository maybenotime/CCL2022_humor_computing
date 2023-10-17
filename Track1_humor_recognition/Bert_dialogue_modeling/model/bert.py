from pickle import TRUE
from turtle import forward
from transformers import BertModel,BertForSequenceClassification
from torch.nn import TransformerEncoder
from torch import nn
import torch

class utter_encoder(nn.Module):
    def __init__(self, bert_model, pad_id):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.pad_id = pad_id

    def forward(self,input):
        '''
        补充上文信息时需要再处理一下token_type_ids
        '''
        attention_mask = input.ne(self.pad_id)      #避免对pad id执行attention
        return_dic = self.bert(input_ids=input, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = return_dic['pooler_output']
        utters_embedding = last_hidden_state
        # last_hidden_state = return_dic['last_hidden_state']
        # sen_embedding_list = []
        # for i in range(input.shape[0]):
        #     sep_index = int(torch.nonzero(input[i]==102).squeeze())     #sep id为102
        #     sen_state = last_hidden_state[i,1:sep_index]
        #     sen_embedding, _ = torch.max(sen_state,dim=0)               #max pooling
        #     sen_embedding_list.append(sen_embedding)
        # utters_embedding = torch.stack(sen_embedding_list, dim=0)    #window_size x 768

        return utters_embedding

class dialogue_modeling(nn.Module):
    def __init__(self, bert_model,dialog_encoder,pad_id):
        super().__init__()
        self.utter_encoder = utter_encoder(bert_model=bert_model,pad_id=pad_id)
        self.d_model = self.utter_encoder.bert.config.hidden_size
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        # self.dialogue_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)
        self.dialogue_encoder = BertForSequenceClassification.from_pretrained(dialog_encoder)
        self.cls = nn.Sequential(nn.Linear(self.d_model, 256),
                            nn.GELU(),
                            nn.Dropout(0.3),
                            nn.Linear(256,64),
                            nn.GELU(),
                            nn.Dropout(0.3),
                            nn.Linear(64,2))
    
    def forward(self,input):
        utters_representation = self.utter_encoder(input)
        dialogue_input = utters_representation.view(-1,4,self.d_model)       #按照窗口为4组织batch
        output = self.dialogue_encoder(inputs_embeds=dialogue_input,output_hidden_states=True)
        last_hidden_sequence = output.hidden_states[0]
        logits = self.cls(last_hidden_sequence)
        logits = logits.view(-1,2)
        return {"pred":logits}

    def predict(self,input):
        utters_representation = self.utter_encoder(input)
        dialogue_input = utters_representation.view(1,-1,self.d_model)       #按照窗口为4组织batch
        output = self.dialogue_encoder(inputs_embeds=dialogue_input,output_hidden_states=True)
        last_hidden_sequence = output.hidden_states[0]
        logits = self.cls(last_hidden_sequence)
        logits = logits.view(-1,2)
        return {"pred":logits}

