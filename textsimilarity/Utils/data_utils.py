import os

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from Utils.data_class import InputExample, InputFeatures

def read_examples(filename):
    with open(filename, 'r') as f:
        lines = [data.strip().split('\t') for data in f.readlines()]
    
    examples = []
    for index,line in enumerate(lines):
        guid = int(line[0]) if line[0].isdigit() else index
        text_a = line[1]
        text_b = line[2]
        label = line[3]

        examples.append(
            InputExample(guid, text_a, text_b, label)
        )
    
    return examples

def load_data(args, tokenizer, data_type):
    if data_type == "train":
        batch_size = args.train_batch_size
        filename = os.path.join(args.data_dir, 'train.tsv')
    elif data_type == "dev":
        batch_size = args.dev_batch_size
        filename = os.path.join(args.data_dir, 'dev.tsv')
    elif data_type == "test":
        batch_size = args.test_batch_size
        filename = os.path.join(args.data_dir, 'test.tsv')
    else:
        raise RuntimeError("should be train or dev or test")

    examples = read_examples(filename)
    features = convert_examples_to_features(examples, args.max_length, tokenizer)
    dataset = convert_features_to_tensors(features, batch_size)
    return dataset


def convert_examples_to_features(examples, max_length, tokenizer):
    label_list = ['0', '1']
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example {} of {}".format(ex_index, len(examples)))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        #padding
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        label_id = label_map[example.label]
        idx = int(example.guid)

        features.append(
            InputFeatures(
                idx=idx, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label_id
            )
        )
    return features
    
def convert_features_to_tensors(features, batch_size):
    all_idx_ids = torch.tensor(
        [f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_idx_ids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def random_dataloader(dataset, batch_size):
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def sequential_dataloader(dataset, batch_size):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
