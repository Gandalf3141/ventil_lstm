#import torch.utils
from torch.utils.data import Dataset, DataLoader

class custom_simple_dataset(Dataset):
 
    def __init__(self, data, window_size):
 
        self.data = data
        self.ws = window_size
 
    def __len__(self):
        return self.data.size(0)
 
    def __getitem__(self, idx):
 
        inp = self.data[idx, :, :]
        label = self.data[idx, self.ws:, 2:]

        return inp, label
