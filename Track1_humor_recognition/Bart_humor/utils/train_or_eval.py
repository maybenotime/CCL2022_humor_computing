import os
import csv
import torch
import fitlog
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

inference_result = './test_result/task1_prob.csv'
logger = logging.getLogger(__name__)


def train(train_dataloader, eval_dataloader, test_dataloader, model, training_args, other_args):
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    best_score = 0

    steps_per_epoch = len(train_dataloader)

    # total number of training steps
    num_train_steps = int(steps_per_epoch * training_args.num_train_epochs)
    t_total = num_train_steps

    no_decay = ["bias", "LayerNorm.weight"]         #这些参数不要做权重衰减，不知道为啥
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_ratio * t_total, num_training_steps=t_total)

    # multi-gpu training
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0

    for epoch in trange(int(training_args.num_train_epochs), desc="Epoch"):

        training_steps = 0
        model.zero_grad()

        for data in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            model.train()
            outputs = model(**data)
            loss, ce_loss, gen_loss = outputs.loss, outputs.ce_loss, outputs.gen_loss

            if training_args.n_gpu > 1:         #多卡时计算平均loss
                loss = loss.mean()

            loss.backward()             #梯度反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)     #梯度裁剪

            optimizer.step()            #做一次梯度下降
            scheduler.step()

            optimizer.zero_grad()       #梯度清0

            training_steps += 1
            global_step += 1

            if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:

                fitlog.add_loss(loss, name="Loss", step=global_step)
                fitlog.add_loss(ce_loss, name="CE_Loss", step=global_step)
                fitlog.add_loss(gen_loss, name="Gen_Loss", step=global_step)

                results = evaluate(training_args, other_args, eval_dataloader, model, "evaluate")
                torch.cuda.empty_cache()
                fitlog.add_metric({"dev": results}, step=global_step)
                if other_args.task_name == 'humor':
                    eval_metrics = 'f1'
                else:
                    print("任务参数输入错误")
                if results[eval_metrics]+results['acc'] > best_score:
                    best_score = results[eval_metrics]+results['acc']
                    fitlog.add_best_metric({"dev": {'best_score': best_score,'f1':results[eval_metrics], 'acc':results['acc']}})

                    # save the best model
                    output_dir = os.path.join(training_args.output_dir, "best_model_%d" % training_args.seed)
                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    results = evaluate(training_args, other_args, test_dataloader, model, "predict")
                    fitlog.add_metric({"test": {'f1': results['f1']},'acc':results['acc']}, step=global_step)

                    fitlog.add_best_metric({"test": {'f1': results['f1'],'acc':results['acc']}})

        torch.cuda.empty_cache()


def evaluate(training_args, other_args, eval_loader, model, eval_or_test):
    def compute_acc_for_categories(preds, labels):
        categories_count = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_right = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_acc = {}
        for pred, label in zip(preds, labels):
            categories_count["label_%s" % label] += 1
            if pred == label:
                categories_right["label_%s" % label] += 1
        for index, (key, value) in enumerate(categories_count.items()):
            categories_acc["label_%s" % index] = format(categories_right["label_%s" % index] / value, '.4f')
        print(categories_acc)
        return categories_acc

    def compute_metrics(preds_id, labels_id):
        results = {}

        # -------------- eval classification --------------
        accuracy = round(accuracy_score(labels_id, preds_id) * 100, 4)
        if other_args.task_name == 'humor':
            f1 = f1_score(labels_id, preds_id, average='binary')          #二分类问题的F1 score
        else:
            print("任务参数输入错误")
        results['acc'] = accuracy_score(labels_id, preds_id)
        results['f1'] = round(f1, 4)

        return results

    results = {}

    if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
        os.makedirs(training_args.output_dir)

    # training_args.eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu eval
    if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running %s *****" % eval_or_test)
    logger.info("  Num examples = %d", len(eval_loader.dataset))
    logger.info("  Batch size = %d", training_args.eval_batch_size)
    # eval_loss = 0.0

    all_preds, all_labels = [], []
    all_probs = []
    for batch in tqdm(eval_loader, desc=eval_or_test):
        model.eval()
        batch = tuple(v.to(training_args.device) for _, v in batch.items())

        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'speakers': batch[3]}
            labels = batch[2]

            labels = labels[labels.ne(-100)].cpu().numpy()

            outputs = model(**inputs)
            preds = F.softmax(outputs.cls_logits)
            probs = preds[:,1]
            probs = probs.cpu().numpy().tolist()
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.extend(probs)
            

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # all_probs = np.concatenate(all_probs, axis=0)
    print(len(all_probs))
    correct_num = np.sum(all_preds == all_labels)

    # eval_loss = eval_loss / nb_eval_steps
    result = compute_metrics(all_preds, all_labels)
    results.update(result)
    logger.info("***** %s results *****" % eval_or_test)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
   
    logger.info("Correct / Total num = ({}/{})".format(correct_num, len(all_labels)))

    csv_data = []
    for label,prob in zip(all_preds,all_probs):
        row = [label,prob]
        csv_data.append(row)

    header = ['Label','Prob']
    with open(inference_result,"w",encoding='utf-8',newline='') as w:
        writer = csv.writer(w)
        writer.writerow(header)
        writer.writerows(csv_data)

    return results
