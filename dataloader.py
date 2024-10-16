#custom dataclass

#This custom dataclass moves a sliding window over a timeseries and returns
# inp(0:t) and label(t+1)
#if values are missing the last value is used for padding sind 
#Structure of the data:

# (number of timeseries // timesteps // features)
# if index+window_size > timesteps  >>> padding

import torch.utils
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):


    def __init__(self, data, window_size, future=1):

        self.data = data
        self.ws = window_size
        self.future = future

    def __len__(self):
        return self.data.size(0)*self.data.size(1) - (self.ws + 1) - (self.future-1)

    def __getitem__(self, idx):

        j = int(idx/self.data.size(1))  

        k = int((idx + self.ws + (self.future-1)) / self.data.size(1))

        m = (idx + self.ws) - k * self.data.size(1)

        index = idx % self.data.size(1)

        if j < k :
            
            if m < 0: 
                inp = self.data[j, index : index + self.ws, :]
            else: 
                inp=torch.cat((self.data[j, index : self.data.size(1) , :],
                          self.data[j, self.data.size(1) - 1, :].repeat(m, 1)))
                
            if self.future>1:
                label = self.data[j, self.data.size(1) - 1, :].repeat(self.future, 1)        
            else:
                label = self.data[j, self.data.size(1) - 1, :]
                
        else:

            inp = self.data[j, index : index + self.ws, :]

            if self.future>1:
                label = self.data[j, index + self.ws : index + self.ws + self.future  , :]
            else:
                label = self.data[j, index + self.ws, :]

        return inp, label

    def get_all_data(self):
        return self.data
    

# We have to include time as an input for neural CDEs !
class CustomDataset_cde(Dataset):


    def __init__(self, data, window_size, future=1):

        self.data = data
        self.ws = window_size
        self.future = future

    def __len__(self):
        return self.data.size(0)*self.data.size(1) - (self.ws + 1) - (self.future-1)

    def __getitem__(self, idx):

        j = int(idx/self.data.size(1))  

        k = int((idx + self.ws + (self.future-1)) / self.data.size(1))

        m = (idx + self.ws) - k * self.data.size(1)

        index = idx % self.data.size(1)

        if j < k :
            
            if m < 0: 
                inp = self.data[j, index : index + self.ws, :]
            else: 
                inp=torch.cat((self.data[j, index : self.data.size(1) , :],
                          self.data[j, self.data.size(1) - 1, :].repeat(m, 1)))
                
            if self.future>1:
                label = self.data[j, self.data.size(1) - 1, :].repeat(self.future, 1)        
            else:
                label = self.data[j, self.data.size(1) - 1, :]
                
        else:

            inp = self.data[j, index : index + self.ws, :]

            if self.future>1:
                label = self.data[j, index + self.ws : index + self.ws + self.future  , :]
            else:
                label = self.data[j, index + self.ws, :]

        return inp, label

    def get_all_data(self):
        return self.data


class custom_simple_dataset(Dataset):
 
 
    def __init__(self, data, window_size):
 
        self.data = data
        self.ws = window_size
 
    def __len__(self):
        return self.data.size(0)
 
    def __getitem__(self, idx):
 
        inp = self.data[idx, :, :]
        label = self.data[idx, self.ws:, 1:].clone()

        return inp, label

class CustomDataset_mlp(Dataset):


    def __init__(self, data, window_size, future=1):

        self.data = data
        self.ws = window_size
        self.future = future

    def __len__(self):
        return self.data.size(0)*self.data.size(1) - (self.ws + 1) - (self.future-1)

    def __getitem__(self, idx):

        j = int(idx/self.data.size(1))  

        k = int((idx + self.ws + (self.future-1)) / self.data.size(1))

        m = (idx + self.ws) - k * self.data.size(1)

        index = idx % self.data.size(1)

        if j < k :
            
            if m < 0: 
                inp = self.data[j, index : index + self.ws, :]
            else: 
                inp=torch.cat((self.data[j, index : self.data.size(1) , :],
                          self.data[j, self.data.size(1) - 1, :].repeat(m, 1)))
                
            if self.future>1:
                label = self.data[j, self.data.size(1) - 1, :].repeat(self.future, 1)        
            else:
                label = self.data[j, self.data.size(1) - 1, :]
                
        else:

            inp = self.data[j, index : index + self.ws, :]

            if self.future>1:
                label = self.data[j, index + self.ws : index + self.ws + self.future  , :]
            else:
                label = self.data[j, index + self.ws, :]

        last = inp[-1:,:]

        inp = torch.cat((inp[:,0], inp[:,1], inp[:,2]))
        
        return inp, last, label
