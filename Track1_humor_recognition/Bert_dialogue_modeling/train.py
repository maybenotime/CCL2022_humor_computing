from fastNLP import CrossEntropyLoss, LossFunc
from fastNLP import SequentialSampler, cache_results, WarmupCallback, GradientClipCallback
from utils.pipe import BertPipe 
from utils.metric import MixMetric
from utils.label_smoothing import label_smoothing, seed_torch, focal_loss
from model.bert import dialogue_modeling
from fastNLP import Trainer
from torch import optim
import wandb
import torch

data_path = './data'

######
lr = 2e-5
n_epochs = 10
bert_model = 'bert-base-uncased'
dialogue_encoder = '/aliyun-06/share_v2/yangshiping/projects/DialogBERT/output/DialogBERT/base/models/checkpoint-200000/context_encoder'
batch_size = 16
up_num =0
seed =2022
######
hyper = {"lr":lr, "bert_model":bert_model, 'dialogue_encoder':dialogue_encoder , "batch_size":batch_size, "up_num":up_num, "seed":seed}
wandb.init(config=hyper, project="bert_dialogue_cls")

seed_torch(seed)
@cache_results('caches/up_{}.pkl'.format(up_num), _refresh=False)        #将预处理的数据缓存
def get_data():
    data_bundle = BertPipe(bert_model, up_num).process_from_file(data_path)
    return data_bundle

data_bundle = get_data()
pad_id = data_bundle.pad_id

model = dialogue_modeling(bert_model=bert_model,dialog_encoder=dialogue_encoder,pad_id=pad_id)

if torch.cuda.is_available():
    print("模型已加载至GPU")
    model.cuda()
else:
    print("未发现可用的计算资源")

optimizer = optim.AdamW(model.parameters(), lr=lr)
sampler = SequentialSampler() 
loss_func = LossFunc(label_smoothing, pred= "pred", target = "label")

train_data = data_bundle.get_dataset('train')
train_data.add_seq_len('input')
dev_data = data_bundle.get_dataset('valid')
test_data = data_bundle.get_dataset('test')

callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.1, schedule='linear')] 


trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=focal_loss(inputs="pred", targets="label"),
                 batch_size=batch_size, sampler=sampler, drop_last=False, update_every=4,
                 num_workers=1, n_epochs=n_epochs, print_every=5,
                 dev_data=dev_data, metrics=MixMetric(),
                 metric_key='mixscore',
                 validate_every=400, save_path='save_models/', use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=-1)
trainer.train(load_best_model=True)