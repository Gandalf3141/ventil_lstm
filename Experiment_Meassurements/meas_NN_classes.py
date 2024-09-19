# NN Classes used for simulation of the full system


import torch
from torch import nn
from torch.nn.utils import weight_norm
#import torchcde

# strg + k + 1/2

#############################################################################################
###                  >>>       OR - derivative prediction       <<<                       ###
#############################################################################################

# OR - LSTM
class OR_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, one_full_traj):

        seq = one_full_traj[:, 0:self.ws, :]
        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)
        #only update next step
        out = one_full_traj[:, self.ws-1:self.ws, 2:] + pred[:, -1: , :]

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

            tmp = torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:2] , out[:, :, :]), dim=2)
            seq = torch.cat((one_full_traj[:, t:self.ws, :], tmp), dim=1)

            lstm_out, hidden = self.lstm(seq)           
            pred = self.linear(lstm_out)
            out = torch.cat((out, out[:, -1:, :] + pred[:, -1: , :]), dim=1)
            
        for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):

            seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:2], out[:, t - self.ws : t , :]), dim=2)
            
            lstm_out, hidden = self.lstm(seq)           
            pred = self.linear(lstm_out)

            out = torch.cat((out, out[:, t-1:t, :] + pred[:, -1: , :]), dim=1)

        return out, hidden 

    def simple_forward(self, seq):

        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)

        return pred, hidden  
 