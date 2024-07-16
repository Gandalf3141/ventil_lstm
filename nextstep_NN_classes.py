import torch
from torch import nn
from torch.nn.utils import weight_norm
import torchcde

# strg + k + 1/2

#############################################################################################
###           >>>       OR - next step prediction       <<<                               ###
#############################################################################################

# OR - LSTM
class LSTMmodel_or_nextstep(nn.Module):

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
        out = pred[:, -1: , :]

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

            tmp = torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:1] , out[:, :, :]), dim=2)
            seq = torch.cat((one_full_traj[:, t:self.ws, :], tmp), dim=1)

            lstm_out, hidden = self.lstm(seq)           
            pred = self.linear(lstm_out)
            out = torch.cat((out, pred[:, -1: , :]), dim=1)
        for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):

            seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:1], out[:, t - self.ws : t , :]), dim=2)
            
            lstm_out, hidden = self.lstm(seq)           
            pred = self.linear(lstm_out)

            out = torch.cat((out, pred[:, -1: , :]), dim=1)
            
        return out, hidden          
  
# Multilayer perceptron
class MLP_or_nextstep(nn.Module):
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh", act_at_end = None, timesteps=5):
        super(MLP_or_nextstep, self).__init__()
        
        if act_fn == "tanh":
            fn = nn.Tanh()
        else:
            fn = nn.ReLU()

        hidden_sizes = [hidden_size for x in range(l_num)]
        # Create a list to hold the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(fn)
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(fn)
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        #Try final non linearity:
        if act_at_end != None:
            if act_at_end == "tanh":
                layers.append(nn.Tanh())
            if act_at_end == "relu":
                layers.append(nn.ReLU())
            if act_at_end == "sigmoid":
                layers.append(nn.Sigmoid())
        
        # Use nn.Sequential to put together the layers
        self.network = nn.Sequential(*layers)
        self.ws = timesteps
        #self.timesteps = timesteps
    
    def forward(self, one_full_traj):
        
        seq = one_full_traj[:, 0:self.ws, :]

        #inp = torch.cat((seq[:, :self.ws,0], seq[:, :self.ws,1], seq[:, :self.ws,2]), dim=2)
        inp = torch.stack([torch.cat((a[:, 0], a[:, 1], a[:, 2])) for a in seq])
        pred = self.network(inp) 
        
        
        out =  pred.view(one_full_traj.size(dim=0),1,2)
        #out = one_full_traj[:, self.ws-1:self.ws, 1:]

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):


            tmp = torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:1] , out[:, :, :]), dim=2)
            seq = torch.cat((one_full_traj[:, t:self.ws, :], tmp), dim=1)

            inp = torch.stack([torch.cat((a[:, 0], a[:, 1], a[:, 2])) for a in seq])

            pred = self.network(inp)

            out = torch.cat((out, pred.view(one_full_traj.size(dim=0),1,2)), dim=1)

        for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):

            seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:1], out[:, t - self.ws : t , :]), dim=2)
            
            inp = torch.stack([torch.cat((a[:, 0], a[:, 1], a[:, 2])) for a in seq])

            pred = self.network(inp)

            out = torch.cat((out, pred.view(one_full_traj.size(dim=0),1,2)), dim=1)

        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):


    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN_or_nextstep(nn.Module):


    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, windowsize=5):
        super(TCN_or_nextstep, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.ws = windowsize

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, one_full_traj):

        
        # war falsch ! (hat trotzdem funktioniert???)
        #seq = one_full_traj[:, 0:self.ws, :]
        seq = one_full_traj[:, :, 0:self.ws]

        y1 = self.tcn(seq)
        pred = self.linear(y1[:, :, -1])
        #only update next step
        out = pred
        out = out.unsqueeze(-1)
        #derivatie_sv = pred

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

            tmp = torch.cat((one_full_traj[:, 0:1, self.ws+(t-1):self.ws+(t-1)+(out.size(dim=2))] , out[:, :, :]), dim=1)
            seq = torch.cat((one_full_traj[:, :, t:self.ws], tmp), dim=2)

            y1 = self.tcn(seq)
            pred = self.linear(y1[:, :, -1])

            next_step = pred
            next_step = next_step.unsqueeze(-1)

            out = torch.cat((out, next_step), dim=2)

            #derivatie_sv = torch.cat((derivatie_sv, pred), dim=2)

        for t in range(self.ws, one_full_traj.size(dim=2) - self.ws):

            seq = torch.cat((one_full_traj[:, 0:1, t : t + self.ws], out[:, :, t - self.ws : t]), dim=1)
            
            y1 = self.tcn(seq)
            pred = self.linear(y1[:, :, -1])

            next_step = pred
            next_step = next_step.unsqueeze(-1)

            out = torch.cat((out, next_step), dim=2)

            #derivatie_sv = torch.cat((derivatie_sv, pred[:, -1: , :]), dim=1)

        return out


#############################################################################################
###               >>>       no OR - derivative prediction       <<<                       ###
#############################################################################################

# Return value is the derivative! addition happens in training / testing function .. why? who knows..

# LSTM
class LSTMmodel_derivative(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):

        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)

        return pred, hidden          

# Multilayer perceptron
class MLP_derivative(nn.Module):
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh", act_at_end = None, timesteps=5):
        super(MLP_derivative, self).__init__()
        
        if act_fn == "tanh":
            fn = nn.Tanh()
        else:
            fn = nn.ReLU()

        hidden_sizes = [hidden_size for x in range(l_num)]
        # Create a list to hold the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(fn)
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(fn)
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        #Try final non linearity:
        if act_at_end != None:
            if act_at_end == "tanh":
                layers.append(nn.Tanh())
            if act_at_end == "relu":
                layers.append(nn.ReLU())
            if act_at_end == "sigmoid":
                layers.append(nn.Sigmoid())
        
        # Use nn.Sequential to put together the layers
        self.network = nn.Sequential(*layers)
        self.ws = timesteps
        #self.timesteps = timesteps
    
    def forward(self, seq):
    
        #inp = torch.stack([torch.cat((a[:, 0], a[:, 1], a[:, 2])) for a in seq])
        # seq is already formatted for mlp input
        pred = self.network(seq) 

        return pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):


    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# TCN
class TCN_derivative(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, windowsize=5):
        super(TCN_derivative, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.ws = windowsize

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, one_full_traj):

        seq = one_full_traj[:, :, 0:self.ws]
        y1 = self.tcn(seq)
        pred = self.linear(y1[:, :, -1])

        return pred


#############################################################################################
###               >>>       no OR - next step prediction       <<<                       ###
#############################################################################################

# LSTM
class LSTMmodel_no_or_nextstep(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):

        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)
        out = pred[:, -1: , :]

        return out, hidden          
  
# Multilayer perceptron
class MLP_no_or_nextstep(nn.Module):
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh", act_at_end = None, timesteps=5):
        super(MLP_no_or_nextstep, self).__init__()
        
        if act_fn == "tanh":
            fn = nn.Tanh()
        else:
            fn = nn.ReLU()

        hidden_sizes = [hidden_size for x in range(l_num)]
        # Create a list to hold the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(fn)
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(fn)
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        #Try final non linearity:
        if act_at_end != None:
            if act_at_end == "tanh":
                layers.append(nn.Tanh())
            if act_at_end == "relu":
                layers.append(nn.ReLU())
            if act_at_end == "sigmoid":
                layers.append(nn.Sigmoid())
        
        # Use nn.Sequential to put together the layers
        self.network = nn.Sequential(*layers)
        self.ws = timesteps
        #self.timesteps = timesteps
    
    def forward(self, seq):
        
        #inp = torch.stack([torch.cat((a[:, 0], a[:, 1], a[:, 2])) for a in seq])
        pred = self.network(seq) 
    
        return pred

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):


    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
#TCN    
class TCN_no_or_nextstep(nn.Module):


    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, windowsize=5):
        super(TCN_no_or_nextstep, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.ws = windowsize

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, one_full_traj):

        seq = one_full_traj[:, :, 0:self.ws]
        y1 = self.tcn(seq)
        pred = self.linear(y1[:, :, -1])
        
        return pred

