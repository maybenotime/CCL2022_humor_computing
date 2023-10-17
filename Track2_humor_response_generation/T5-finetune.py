import torch
import json
import os 
import wandb
import argparse
from data.dataset import DialogData
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback



save_model_dir = './true_large_checkpoints'
train_humor_data = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/humor_data/train.json'
dev_humor_data = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/humor_data/task2_dev.json'
dailydialog_path = '/aliyun-06/share_v2/yangshiping/projects/humor_computation/T5_humor/data/dailydialog.json'

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str, default=save_model_dir)   #保存模型的路径
    parser.add_argument("--train_data_path", type=str, default=train_humor_data)
    parser.add_argument("--dev_data_path", type=str, default=dev_humor_data)
    parser.add_argument("--dailydialog_path", type=str, default=dailydialog_path)
    parser.add_argument("--batch_size_on_train", type=int, default=4)
    parser.add_argument("--batch_size_on_eval", type=int, default=4)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)     #跑几个epoch
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=20)         #几个epoch loss没有下降则停止训练
    parser.add_argument("--wandb_running_name", type=str, default='test_mt5')
    
    args = parser.parse_args()
    return args

def load_humor_json(daily_path,train_path,dev_path):        #加载数据
    with open(daily_path,"r") as f_o:       #额外引入的dailydialog数据
        daily_dialog = json.load(f_o)

    with open(train_path,"r") as f_t:
        train_data = json.load(f_t)

    with open(dev_path,"r") as f_d:
        dev_data = json.load(f_d)
    
    return daily_dialog,train_data,dev_data


def main(args):
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-large')
    tokenizer.truncation_side = 'left'      #我们可以截断离目标回复最远的context
    t5model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large')
    daily_dialog,humor_train,humor_valid = load_humor_json(args.dailydialog_path,args.train_data_path,args.dev_data_path)
    train_dataset = DialogData(daily_dialog,humor_train,tokenizer)
    valid_dataset = DialogData(None,humor_valid,tokenizer)
    wandb.init(project="t5-humor_generate")
    
    #配置trainer参数
    args = Seq2SeqTrainingArguments(                        
        output_dir=args.output_model_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_on_train,
        per_device_eval_batch_size=args.batch_size_on_eval,
        weight_decay=0.01,
        save_total_limit=10,                #checkpoints中最多会保留几个模型
        local_rank = int(os.environ.get('LOCAL_RANK')),     
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        save_strategy='epoch', 
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',         #验证集的loss最低时，train上loss尚未收敛，所以取消早停策略
        run_name=args.wandb_running_name,
        report_to='wandb',                      #报告结果和日志的平台
        logging_dir='../logs',
        generation_max_length=128,
        generation_num_beams=10,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer)  #类似于pytorch中的dataloader
    #EarlyStoppingCallback(early_stopping_patience=args.patience)   #早停策略

    trainer = Seq2SeqTrainer(
        t5model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()         #进行微调




if __name__ == '__main__':
    arg = args()
    main(arg)
    

