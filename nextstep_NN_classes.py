import torch
from torch import nn
from torch.nn.utils import weight_norm
import torchcde

# OR - LSTM
class LSTMmodel(nn.Module):

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
class OR_MLP(nn.Module):
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh", act_at_end = None, timesteps=5):
        super(OR_MLP, self).__init__()
        
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
        print(out.size(),out)

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
    
class OR_TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, windowsize=5):
        super(OR_TCN, self).__init__()
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

######################
# A CDE model looks like

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_width=128):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_width  = hidden_width

        self.linear1 = torch.nn.Linear(hidden_channels, self.hidden_width)
        self.linear2 = torch.nn.Linear(self.hidden_width, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z
######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_width, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, hidden_width)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval, backend='torchdiffeq', options=dict(jump_t=X.grid_points),)
                                #method='rk4', 
                                #options=dict(step_size=2e-4),
                                
                                #adjoint=False, atol = 1e-4, rtol = 1e-4)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y