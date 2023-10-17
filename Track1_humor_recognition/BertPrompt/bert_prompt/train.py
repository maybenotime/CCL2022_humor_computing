from openprompt.data_utils import InputExample
from openprompt.prompts import MixedTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score 
import os
import torch
import json
import wandb
import random
import numpy as np

###### hyper_parameters######
lr = 2e-5
n_epochs = 10
bert_model = 'bert-base-uncased'
batch_size = 16
seed =2022
#############################
hyper = {"lr":lr, "bert_model":bert_model, "batch_size":batch_size,"seed":seed}
wandb.init(config=hyper, project="bert_prompt_cls")

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(seed)
data_dir = './data'

dataset = {}
file_list = ['train.json','valid.json','test.json']
for file in file_list:
    path = os.path.join(data_dir, file)
    with open(path, 'r') as f:
        data = json.load(f)
        dataset[file.split('.')[0]] = []
        for dialogue in data:
            last_utter = ''
            for utter in dialogue:
                sen = utter["Sentence"]
                speaker = utter["Speaker"]
                sen_plus = speaker + ':' + sen
                meta = dict(speaker=speaker)
                input_example = InputExample(text_a = sen_plus, text_b = last_utter, label=int(float(utter["Label"])), meta=meta)
                last_utter = sen_plus
                dataset[file.split('.')[0]].append(input_example)

classes = ["not humorours","humorous"]      #类别
plm, tokenizer, model_config, WrapperClass = load_plm("bert", bert_model)

template_text = 'a {"mask"} utterance: {"placeholder":"text_a"}'     #定义模板
#'a {"mask"} utterance: {"placeholder":"text_a"}'      该模板 1.0638
#'{"placeholder":"text_a"} It is a {"mask"} utterance.' 该模板 1.0579
mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
# print(wrapped_example)
myverbalizer = ManualVerbalizer(                                            #定义映射
    tokenizer,
    classes=classes,
    label_words={
        "humorous":["humorous","humor","funny"],
        "not humorours":["simple","nature","ordinary"]
    })

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Now the training is standard
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["valid"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

best_score = 0
for epoch in range(n_epochs):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        wandb.log({'loss':loss})
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, step: {}".format(epoch,step))
    allpreds = []
    alllabels = []
    prompt_model.eval()
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = accuracy_score(alllabels, allpreds)
    f1 = f1_score(alllabels, allpreds, average='binary')
    recall = recall_score(alllabels, allpreds, average='binary')
    pre = precision_score(alllabels, allpreds, average='binary')
    mixscore = acc + f1
    if mixscore > best_score:
        best_score = mixscore
        wandb.run.summary["best_mixscore"] = best_score
    result = {"mixscore":mixscore, "f1":f1, "acc":acc, "recall":recall, "precision":pre}
    wandb.log(result)
    print(result)

print(best_score)
