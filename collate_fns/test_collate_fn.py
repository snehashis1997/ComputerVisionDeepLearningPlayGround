import numpy as np
import torch

sents = ["Hi there", "hello", "hello guys", "I am Snehashis"]
unique_words = list(set([word.lower() for item in sents for word in item.split(" ")]))

# print(unique_words)
n_unique_words = len(unique_words)
embs = torch.randn((n_unique_words,4))

mapping = dict(zip(unique_words, embs))

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, sents, mapping):
        self.sents = sents
        self.mapping = mapping

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent = self.sents[idx]
        words = sent.split(" ")
        temp = []
        for word in words:
            temp.append(mapping[word.lower()])
        temp = torch.vstack(temp)
        return temp
    

def custom_collate(data):
    max_dim = data[0].shape[0]
    for item in data:
        print(item.shape)
        if max_dim < item.shape[0]:
            max_dim = item.shape[0]
    
    print(max_dim)
    t = []
    for item in data:
        if item.shape[0] != max_dim:
            padded_tensor = torch.zeros((max_dim - item.shape[0], 4))
            resulted_padded_tensor = torch.vstack([item, padded_tensor])
            t.append(resulted_padded_tensor)
            # print(resulted_padded_tensor.shape)
        else:
            t.append(item)
    t = torch.vstack(t)
    print(t.shape)
    return t

    # pass

dl_with_collate = torch.utils.data.DataLoader(CustomDataset(sents, mapping),
                                              batch_size = 3,
                                              shuffle = False,
                                              collate_fn = custom_collate)

it = iter(dl_with_collate)
it.__next__()