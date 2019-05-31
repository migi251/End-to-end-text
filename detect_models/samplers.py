from torch.utils.data.sampler import RandomSampler,Sampler
import numpy as np
import copy
import random

def build_train_sampler(data_source,
                        train_sampler,
                        **kwargs):
    """Build sampler for training

    Args:
    - data_source (list): list of (img,...).
    """
    if train_sampler=='RandomSampler':
        sampler = RandomSampler(data_source)
    elif train_sampler =='LOLSampler':
        sampler = LOLSampler(data_source)
    else:
        print('Suport RandomSampler only!')

    return sampler


class LOLSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.length = len(self.data_source)*2

    def __iter__(self):
        final_idxs = []
        for i in range(len(self.data_source)):
            final_idxs.append(i)
            final_idxs.append(i)

        

        return iter(final_idxs)

    def __len__(self):
        return self.length