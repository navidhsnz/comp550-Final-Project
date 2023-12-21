

import random
import torch


# Maybe a better way to do this
class DataLoader:
    def __init__(self, data, batch_size, truncate=True):
        # data shoud be dict of language pairs (keys tuple of languages, v tuple of exampels)

        self.batch_size = batch_size
        self.data = data

        # Shuffle data
        for k, v in data.items():
            l1, l2 = v
            
            indices = torch.randperm(l1.size()[0])

            data[k] = (l1[indices], l2[indices])

        
        if truncate:
            min_size = min([d[0].shape[0] for d in data.values()])
            for k, v in data.items():
                l1, l2 = v
                data[k] = (l1[:min_size], l2[:min_size])


        # for deterministic batching
        self.r = random.Random(42)

        self.pairs = list(data.keys())

        self.langs = set()
        self.langs.update(*self.pairs)

        self.num_langs = len(self.langs)

        self.nums_examples = {k: len(v[0]) for k, v in data.items()}
        
        self.num_batches = sum(self.nums_examples.values()) // self.batch_size + 1

                

    def __iter__(self):
            
        # have to change to allow for pairs to be given in either direction (swap src and tgt langs)

        cur_pairs = [p for p in self.pairs]
        prev_inds = {k: 0 for k in self.data.keys()}

        while len(cur_pairs) > 0:
            langs = self.r.choice(cur_pairs)
            
            prev_ind = prev_inds[langs]
            max_ind = self.nums_examples[langs]
            next_ind = min(prev_ind + self.batch_size, max_ind)

            prev_inds[langs] = next_ind

            if next_ind >= max_ind:
                cur_pairs.remove(langs)

            tup = self.data[langs]

            yield langs[0], langs[1], tup[0][prev_ind:next_ind], tup[1][prev_ind:next_ind]


