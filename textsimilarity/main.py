#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

from config import get_args
from train_eval import train, evaluate
from Utils.utils import get_device, set_seed
from Utils.data_utils import load_data, random_dataloader, sequential_dataloader

from transformers import (
    WEIGHTS_NAME, AdamW,
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    BertConfig, BertForSequenceClassification, BertTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}


def main(args):
    
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
        and args.do_train):
        print("输出目录 ({}) 已经存在且不为空. ".format(args.output_dir))
        print("你想覆盖掉该目录吗？type y or n")

        if input() == 'n':
            return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # gpu ready
    gpu_ids = [int(device_id) for device_id in args.gpu_ids.split()]
    args.device, args.n_gpu = get_device(gpu_ids[0])

    # PTM ready
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_file,
        num_labels = 2, 
        cache_dir=None
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        cache_dir=None
    )

    # train and eval get the checkpoint
    if args.do_train:
        train_dataset = load_data(args, tokenizer, 'train')
        train_dataloader = random_dataloader(train_dataset, args.train_batch_size)

        dev_dataset = load_data(args, tokenizer, 'dev')
        dev_dataloader = sequential_dataloader(dev_dataset, args.dev_batch_size)

        # 模型准备
        model = model_class.from_pretrained(
            args.model_file,
            from_tf=False,
            config=config,
            cache_dir=None
        )
        
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        # optimizer ready
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        train(args, train_dataloader, dev_dataloader, model, optimizer, scheduler, tokenizer)

    # Predict checkpoint result
    
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    test_dataset = load_data(args, tokenizer, 'test')
    test_dataloader = sequential_dataloader(test_dataset, args.test_batch_size)

    model = model_class.from_pretrained(args.output_dir)
    model.to(args.device)

    eval_loss, eval_metric = evaluate(args, model, test_dataloader, do_predict=True)
    for key, val in eval_metric.items():
        print('the test dataset {} is {}'.format(key, val))
    

if __name__ == "__main__":
    args = get_args()
    main(args)
    



