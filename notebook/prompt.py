import logging
import re
import os
# os.chdir('../code/')
import random

import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm, trange
from tqdm.notebook import tqdm, trange
from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoTokenizer, get_linear_schedule_with_warmup)


from ..code.constants import *
from ..code.data_utils import ABSADataset, filter_none, filter_invalid, get_dataset, get_inputs, normalize_augment
from ..code.model_utils import (prepare_constrained_tokens, prepare_tag_tokens)
from ..code.main import *
from ..code.data_utils import *


import json
import argparse
import pandas as pd
from pythainlp import word_tokenize



arg_path = '../outputs/aste/cross_domain/run_aste/seed-42/laptop14-rest14/args.json'
with open(arg_path, 'r') as file:
    args_dict = json.load(file)

# Create a namespace from the dictionary
args = argparse.Namespace(**args_dict)

# args.model_name_or_path = '../outputs/aste/cross_domain/run_aste/seed-42/laptop14-rest14/checkpoint-e24'
args.model_name_or_path = '../outputs/aste/cross_domain/run_aste/seed-42/laptop14-rest14/extract_aste-model'



model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

data = pd.read_csv('../thai_data/wongnai/w_review_train.csv', sep=';',  header=None, )
data.columns = ['review', 'rating']


tqdm.pandas(desc='tokenizing')
data['tokens'] = data['review'].progress_apply(lambda x: word_tokenize(x, engine='newmm'))
data['n_tokens'] = data['tokens'].apply(lambda x:len(x))
data['text_space'] = data['tokens'].apply(lambda x: ' '.join(x))


input_list = data['text_space'].tolist()
dataset = ABSADataset(args, tokenizer, inputs=input_list, targets=[ " " for _ in range(len(input_list))])


def infer_new(args, dataset, model, tokenizer, name=None, is_constrained=False, constrained_vocab=None, keep_mask=False, **decode_dict):
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=4)

    if keep_mask:
        # can't skip special directly, will lose extra_id
        unwanted_tokens = [tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]
        unwanted_ids = tokenizer.convert_tokens_to_ids(unwanted_tokens)
        def filter_decode(ids):
            ids = [i for i in ids if i not in unwanted_ids]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            sentence = tokenizer.convert_tokens_to_string(tokens)
            return sentence

    # inference
    inputs, outputs, targets = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating'):
            if is_constrained:
                prefix_fn_obj = Prefix_fn_cls(tokenizer, constrained_vocab, batch['source_ids'].to(args.device))  # need fix
                prefix_fn = lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
            else:
                prefix_fn = None

            outs_dict = model.generate(input_ids=batch['source_ids'].to(args.device),
                                        attention_mask=batch['source_mask'].to(args.device),
                                        max_length=128,
                                        prefix_allowed_tokens_fn=prefix_fn,
                                        output_scores=True,
                                        return_dict_in_generate=True,
                                        **decode_dict,
                                        )
            outs = outs_dict["sequences"]

            if keep_mask:
                input_ = [filter_decode(ids) for ids in batch["source_ids"]]
                dec = [filter_decode(ids) for ids in outs]
                target = [filter_decode(ids) for ids in batch["target_ids"]]
            else:
                input_ = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
                dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

            inputs.extend(input_)
            outputs.extend(dec)
            targets.extend(target)

    # decode_txt = "constrained" if is_constrained else "greedy"
    # with open(os.path.join(args.inference_dir, f"{name}_{decode_txt}_output.txt"), "w") as f:
    #     for i, o in enumerate(outputs):
    #         f.write(f"{inputs[i]} ===> {o}\n")

    # return inputs, outputs, targets
    return inputs, outputs


input_infer, output_infer = infer_new(
        args, dataset, model, tokenizer, 
        # name=f"thai-pred",
        # is_constrained=True, 
        is_constrained=False, 
        constrained_vocab=prepare_constrained_tokens(tokenizer, args.task, args.paradigm),
    )



with open('../thai_data/wongnai/rest14/w_review_train_extract_v2.csv', "w") as f:
    for x,y in zip(input_infer, output_infer):
        f.write(f"{x} ===> {y}\n")

    