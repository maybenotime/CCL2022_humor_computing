from fastNLP import CrossEntropyLoss, LossFunc
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback, SequentialSampler
from utils.pipe import BertPipe 
from utils.metric import MixMetric
from utils.callbacks import FGM_callback            #对抗训练
import argparse
from utils.label_smoothing import label_smoothing, seed_torch, focal_loss
from model.bert import bertcls
from fastNLP import Trainer
from torch import optim
import wandb
import torch

data_path = './data'
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--warmup_sche", type=str)
    parser.add_argument("--update_every", type=int)
    parser.add_argument("--loss_name", type=int)      
    args = parser.parse_args()
    return args

arg = args()
######
lr = arg.lr
n_epochs = 10
bert_model = 'microsoft/deberta-v3-large'
batch_size = 16
up_num = 3
update_every = arg.update_every
warmup_ratio = arg.warmup_ratio
warmup_sche = arg.warmup_sche
loss_name = arg.loss_name
seed = 2022
#######
all_hyper = {"lr":lr, "bert_model":bert_model, "batch_size":batch_size, "up_num":up_num, "seed":seed}
wandb.init(config=all_hyper, project="bertcls_hyper_opt")

# seed_torch(seed)
@cache_results('caches/up_{}.pkl'.format(up_num), _refresh=False)        #将预处理的数据缓存
def get_data():
    data_bundle = BertPipe(bert_model, up_num).process_from_file(data_path)
    return data_bundle

data_bundle = get_data()
pad_id = data_bundle.pad_id

model = bertcls(bert_model=bert_model, pad_id=pad_id)

if torch.cuda.is_available():
    print("模型已加载至GPU")
    model.cuda()
else:
    print("未发现可用的计算资源")

optimizer = optim.AdamW(model.parameters(), lr=lr)
sampler = BucketSampler()               #sampler对训练过程影响很大
if loss_name == 1:
    loss_func = LossFunc(label_smoothing, pred= "pred", target = "label")       #label_smoothing
elif loss_name == 2:
    loss_func = CrossEntropyLoss(target="label")
elif loss_name == 3:
    loss_func = focal_loss(inputs="pred", targets="label")

train_data = data_bundle.get_dataset('train')
print(train_data[:10])
train_data.add_seq_len('input')
dev_data = data_bundle.get_dataset('valid')
test_data = data_bundle.get_dataset('test')

# callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=0.05, schedule='linear'),FGM_callback()] 
callbacks = [GradientClipCallback(clip_type='value'), WarmupCallback(warmup=warmup_ratio, schedule=warmup_sche)] 
#focal_loss(inputs="pred", targets="label")
#CrossEntropyLoss(target="label")
trainer = Trainer(train_data=train_data, model=model,
                  optimizer=optimizer, loss=loss_func,
                 batch_size=batch_size, sampler=sampler, drop_last=False, update_every=update_every,
                 num_workers=4, n_epochs=n_epochs, print_every=5,
                 dev_data=dev_data, metrics=MixMetric(),
                 metric_key='mixscore',
                 validate_every=200, save_path='save_models/', use_tqdm=True, device=None,
                 callbacks=callbacks, check_code_level=0)
trainer.train(load_best_model=True)


