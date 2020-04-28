#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import time

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from Utils.utils import compute_metrics, softmax

def train(args, train_dataloader, dev_dataloader, model, optimizer, scheduler, tokenizer):

    writer = SummaryWriter(log_dir=args.log_dir + os.sep +
                           time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime(time.time())))

    global_step = 0

    best_dev_loss = float('inf')

    epoch_loss = 0.0
    logging_loss = 0.0
    train_step = 0

    for epoch in range(int(args.num_train_epochs)):
        print('---------------- Epoch: {}s start ----------'.format(epoch))

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[1], "attention_mask": batch[2], "labels": batch[4]}
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in [
                    "bert", "xlnet", "albert"] else None
            )
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            train_step += 1

            epoch_loss += loss.item()
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            label_ids = batch[4].to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)    
            all_preds = np.append(all_preds, preds)

            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step != 0 and global_step % args.save_step == 0:
                    # 保存日志，用 tensorboard 分析
                    dev_loss, dev_metric = evaluate(args, model, dev_dataloader)
                    train_metric = compute_metrics(all_preds, all_labels)               

                    train_loss = (epoch_loss-logging_loss) / train_step
                    learn_rate = scheduler.get_lr()[0]

                    logs = {}
                    logs['loss'+ os.sep +'train'] = train_loss
                    logs['loss' + os.sep + 'dev'] = dev_loss
                    logs['learning_rate'] = learn_rate

                    for key, val in train_metric.items():
                        logs[key+os.sep+'train'] = val

                    for key, val in dev_metric.items():
                        logs[key+os.sep+'dev'] = val

                    for key, val in logs.items():
                        writer.add_scalar(key,val, global_step//args.save_step)
                    
                    # save the checkpoint using  best dev-loss
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        output_dir = args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(
                            output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))


    writer.close()
    return global_step, epoch_loss/global_step

def evaluate(args, model, dataloader, do_predict=False):
    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_idxs = np.array([], dtype=int)
    all_confidences = []

    eval_loss = 0.0
    eval_step = 0

    for batch in tqdm(dataloader, desc='Eval'):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": batch[4]}
            inputs["token_type_ids"] = (
                batch[3] if args.model_type in [
                    "bert", "xlnet", "albert"] else None
            )
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        eval_step += 1

        idxs = batch[0].detach().cpu().numpy()
        confidences = logits.detach().cpu().numpy()
        preds = np.argmax(confidences, axis=1)

        confidences = confidences.tolist()
        all_confidences.extend(confidences)
        
        labels = batch[4].detach().cpu().numpy()

        all_preds = np.append(all_preds, preds)
        all_labels = np.append(all_labels, labels)
        all_idxs = np.append(all_idxs, idxs)
    
    eval_loss = eval_loss / eval_step
    if do_predict:
        eval_result_file = os.path.join(args.output_dir, 'eval_results.txt')
        all_confidences = [softmax(x)[-1] for x in all_confidences]
        with open(eval_result_file, 'w') as f:
            for i in range(len(all_idxs)):
                f.write(str(all_idxs[i]) + '\t' + str(all_preds[i]) + '\t' + str(all_labels[i]) + '\t' + str(all_confidences[i]) + '\n')

    metrics = compute_metrics(all_labels, all_preds)
    return eval_loss, metrics





