
import os

from huggingface_hub.repocard import model_index_to_eval_results
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"

from datasets import load_dataset

import torch
import random



def load_un_pc(languages, tokenizers, num_items=-1, start_item=0, min_len=250, do_reverse=True, swap_pairs=False, self_map=True, short_prob=0.1):

    dataset = {}

    print(languages)
        
    for l1, l2 in languages:

        data = load_dataset("un_pc", l1 + "-" + l2, split="train[" + str(start_item) + ":" + str(num_items) + "]", num_proc=4)
        
        def preprocess_function(examples):
            inputs = [example[l1] for example in examples["translation"]]
            targets = [example[l2] for example in examples["translation"]]
            tok_inputs = tokenizers[l1](inputs, max_length=256, truncation=True, padding="max_length")
            tok_targets = tokenizers[l2](targets, max_length=256, truncation=True, padding="max_length")
            #print(tokenizer.batch_decode(model_inputs["input_ids"]))
            preprocessed = {}
            preprocessed["input_ids"] = tok_inputs["input_ids"]
            preprocessed["labels"] = tok_targets["input_ids"]

            return preprocessed

        data = data.filter(lambda example: len(example["translation"][l1]) >= min_len or len(example["translation"][l1]) < min_len and random.random() < short_prob)

        tokenized = data.map(preprocess_function, batched=True)

        l1_ids = [t["input_ids"] for t in tokenized]
        l2_ids = [t["labels"] for t in tokenized]

        l1_ids = torch.tensor(l1_ids)
        l2_ids = torch.tensor(l2_ids)

        if swap_pairs:
            l1, l2 = l2,l1
            l1_ids, l2_ids = l2_ids, l1_ids

        dataset[(l1, l2)] = (l1_ids, l2_ids)

        if do_reverse:
            dataset[(l2, l1)] = (l2_ids, l1_ids)

        if self_map:
            if (l1, l1) in dataset:
                s, t = dataset[(l1, l1)]
                new_ids = torch.cat((s, l1_ids), dim=0) 
                dataset[(l1, l1)] = (new_ids, new_ids)
            else:
                dataset[(l1, l1)] = (l1_ids, l1_ids)

            if (l2, l2) in dataset:
                s, t = dataset[(l2, l2)]
                new_ids = torch.cat((s, l2_ids), dim=0) 
                dataset[(l2, l2)] = (new_ids, new_ids)
            else:
                dataset[(l2, l2)] = (l2_ids, l2_ids)



    
    return dataset
        







