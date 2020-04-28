#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

def get_args():

    parser = argparse.ArgumentParser(description='BERT Baseline')

    parser.add_argument("--model_name", 
                        default="BertOrigin", 
                        type=str, 
                        help="the name of model")

    parser.add_argument("--model_type",
                        default='bert',
                        type=str,
                        help="PTM 基础模型: bert,albert,roberta,xlnet")

    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--data_dir",
                        default='/ssd2/songyingxin/songyingxin/dataset/QQP',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--output_dir",
                        default="BertOrigin",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # parser.add_argument("--cache_dir", 
    #                     default='cache',
    #                     type=str,
    #                     help="缓存目录，主要用于模型缓存")
    
    parser.add_argument("--log_dir",
                        default='log' + os.sep + 'BertOrigin',
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")

    # PTM 相关文件参数
    parser.add_argument("--config_file",
                        default='/ssd2/songyingxin/PreTrainedModels/bert-uncased-base/bert_config.json',
                        type=str)
    parser.add_argument("--vocab_file",
                        default='/ssd2/songyingxin/PreTrainedModels/bert-uncased-base/bert-base-uncased-vocab.txt',
                         type=str)
    parser.add_argument("--model_file",
                        default='/ssd2/songyingxin/PreTrainedModels/bert-uncased-base/pytorch_model.bin',
                         type=str)

    # 文本预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_length",
                        default=50,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="训练的 epoch 数目")
    parser.add_argument("--learning_rate", 
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")


    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="梯度累积")

    parser.add_argument('--save_step',
                        type=int,
                        default=1000,
                        help="多少步进行模型保存以及日志信息写入")

    parser.add_argument("--gpu_ids", 
                        type=str, 
                        default="0", 
                        help="gpu 的设备id")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="随机种子 for initialization")

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    get_args()
    
