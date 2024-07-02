import torch
from torch import nn
from torch.nn.utils import weight_norm
import torchcde


# OR - LSTM
class LSTMmodel(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4, stepsize=1, rungekutta=False):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size
        self.rungekutta = rungekutta
        if stepsize==1:
         self.step_size = 1
        else:
         self.step_size = torch.nn.parameter.Parameter(torch.rand(1))

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, one_full_traj):

        seq = one_full_traj[:, 0:self.ws, :]
        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)
        #only update next step
        out = one_full_traj[:, self.ws-1:self.ws, 1:] + pred[:, -1: , :]

        derivatie_sv = pred[:, -1: , :]
        #out = one_full_traj[:, self.ws-1:self.ws, 1:] + pred

        if self.rungekutta == False:
            for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

                tmp = torch.cat(( one_full_traj[:,self.ws+t:self.ws+t+1, 0:1] , out[:, (t-1):t,:]), dim=2)
                seq = torch.cat((one_full_traj[:, t:self.ws+(t-1), :], tmp), dim=1)
                lstm_out, hidden = self.lstm(seq)           
                pred = self.linear(lstm_out)
                out = torch.cat((out, one_full_traj[:, self.ws+(t-1): self.ws+t, 1:] + pred[:, -1: , :]), dim=1)

                derivatie_sv = torch.cat((derivatie_sv, pred[:, -1: , :]), dim=1)


            for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):

                seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:1], out[:, t - self.ws : t , :]), dim=2)
                
                lstm_out, hidden = self.lstm(seq)           
                pred = self.linear(lstm_out)

                out = torch.cat((out, out[:, t-1:t, :] + self.step_size * pred[:, -1: , :]), dim=1)

                derivatie_sv = torch.cat((derivatie_sv, pred[:, -1: , :]), dim=1)

        if  self.rungekutta == True:
            for t in range(1, self.ws + 2): # für RK : range(1, self.ws + 2):

                tmp = torch.cat(( one_full_traj[:,self.ws+t:self.ws+t+1, 0:1] , out[:, (t-1):t,:]), dim=2)
                seq = torch.cat((one_full_traj[:, t:self.ws+(t-1), :], tmp), dim=1)
                lstm_out, hidden = self.lstm(seq)           
                pred = self.linear(lstm_out)
                out = torch.cat((out, one_full_traj[:, self.ws+(t-1): self.ws+t, 1:] + pred[:, -1: , :]), dim=1)
            
            for t in range(self.ws, one_full_traj.size(dim=1) - self.ws - 2):
                # seq = torch.cat((out[:, t - self.ws : t , :], one_full_traj[:, t : t + self.ws, 0:1]), dim=2)
                
                # lstm_out, hidden = self.lstm(seq)           
                # pred = self.linear(lstm_out)

                # out = torch.cat((out, out[:, t-1:t, :] + pred[:, -1: , :]), dim=1)

                #Runge Kutta : 
                
                #y(n+1) = y(n) + h/6 * (k1 + 2k2 + 2k3 + k4)
                # k1 = f(y(n))          --- u1
                # k2 = f(y(n)+h/2*k1)   --- u2
                # k3 = f(y(n)+h/2*k2)   --- u2
                # k4 = f(y(n)+h*k3)     --- u3
                # We only have u at discrete steps -> use h = 2 such that y(n)+k1 = y(n+1) and so on

                #richtig wäre :
                seq = torch.cat((one_full_traj[:, t : t + self.ws + 2, 0:1], out[:, t - self.ws : t + 2 , :]), dim=2)
                #war falsch beim training!
                #seq = torch.cat((out[:, t - self.ws : t + 2 , :], one_full_traj[:, t : t + self.ws + 2, 0:1]), dim=2)

                inp1 = seq[:, 0:-2, :]

                lstm_out, hidden = self.lstm(inp1)           
                k1 = self.linear(lstm_out)

                inp2 = seq[:, 1:-1, :]
                #inp2 = seq[:, 0:-2, :]
                #inp2[:, :, 0:1] =  1/2 * (seq[:, 0:-2, 0:1] + seq[:, 1:-1, 0:1])
                inp2[:, -1:, 1:] = inp2[:, -1:, 1:] + k1[:, -1, :]

                lstm_out, hidden = self.lstm(inp2)           
                k2 = self.linear(lstm_out) 

                inp3 = seq[:, 1:-1, :]
                #inp3 = seq[:, 0:-2, :]
                #inp3[:, :, 0:1] =  1/2 * (seq[:, 0:-2, 0:1] + seq[:, 1:-1, 0:1])
                inp3[:, -1:, 1:] = inp3[:, -1:, 1:] + k2[:, -1, :]

                lstm_out, hidden = self.lstm(inp3)           
                k3 = self.linear(lstm_out)          

                #inp4 = seq[:, 1:-1, :]
                inp4 = seq[:, 2:, :]
                inp4[:, -1:, 1:] = inp4[:, -1:, 1:] + 2*k3[:, -1, :]

                lstm_out, hidden = self.lstm(inp4)           
                k4 = self.linear(lstm_out)  

                    # y(n+1) ist 2 steps in der zukunft wegen h = 2 ?!?
                res = out[:, t:t+1, :]  +  2/6 * (k1[:, -1, :] + 2*k2[:, -1, :] + 2*k3[:, -1, :] + k4[:, -1, :])
                out = torch.cat((out, res), dim=1)

            
        return out, hidden, derivatie_sv          
  
# LSTM
class LSTMmodel_nextstep(nn.Module):



    def __init__(self, input_size, hidden_size, out_size, layers):
        """
        Initialize the LSTM model.

        Args:
        - input_size: Size of input
        - hidden_size: Size of hidden layer
        - out_size: Size of output
        - layers: Number of layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.act = nn.ReLU()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):
        """
        Forward pass through the LSTM model.

        Args:
        - seq: Input sequence

        Returns:
        - pred: Model prediction
        - hidden: Hidden state
        """
        lstm_out, hidden = self.lstm(seq)
        #lstm_out = self.act(lstm_out)
        pred = self.linear(lstm_out)

        return pred, hidden

# GRU 
class GRUmodel(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4, stepsize=1):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size

        if stepsize==1:
         self.step_size = 1
        else:
         self.step_size = torch.nn.parameter.Parameter(torch.rand(1))

        # Define LSTM layer
        self.GRU = nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, one_full_traj):

        seq = one_full_traj[:, 0:self.ws, :]
        lstm_out, hidden = self.GRU(seq)           
        pred = self.linear(lstm_out)
        out = one_full_traj[:, self.ws-1:self.ws, 1:] + self.step_size * pred[:, -1: , :]

        for t in range(1, self.ws):

            tmp = torch.cat(( one_full_traj[:,self.ws+t:self.ws+t+1, 0:1] , out[:, (t-1):t,:]), dim=2)
            seq = torch.cat((one_full_traj[:, t:self.ws+(t-1), :], tmp), dim=1)
            lstm_out, hidden = self.GRU(seq)           
            pred = self.linear(lstm_out)
            out = torch.cat((out, one_full_traj[:, self.ws+(t-1): self.ws+t, 1:] + self.step_size * pred[:, -1: , :]), dim=1)

        for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):
            seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:1], out[:, t - self.ws : t , :]), dim=2)
            
            lstm_out, hidden = self.GRU(seq)           
            pred = self.linear(lstm_out)

            out = torch.cat((out, out[:, t-1:t, :] + self.step_size * pred[:, -1: , :]), dim=1)


        return out, hidden          

# Multilayer perceptron
class MLP(nn.Module):
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh", act_at_end = None):
        super(MLP, self).__init__()
        
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
    
    def forward(self, x):
        
        return self.network(x)         



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
    

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


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

        #print("This arrives at the forward pass", one_full_traj[:,:, 0:10])
        
        # war falsch ! (hat trotzdem funktioniert???)
        #seq = one_full_traj[:, 0:self.ws, :]
        seq = one_full_traj[:, :, 0:self.ws]

        y1 = self.tcn(seq)
        pred = self.linear(y1[:, :, -1])
        #only update next step
        out = one_full_traj[:, 1:, self.ws-1] + pred
        out = out.unsqueeze(-1)
        #derivatie_sv = pred

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

            tmp = torch.cat(( one_full_traj[:, 0:1, self.ws+t:self.ws+t+1] , out[:,:,-1:]), dim=1)
            seq = torch.cat((one_full_traj[:, :, t:self.ws+(t-1)], tmp), dim=2)

            y1 = self.tcn(seq)
            pred = self.linear(y1[:, :, -1])

            next_step = one_full_traj[:, 1:, self.ws+(t-1)] + pred
            next_step = next_step.unsqueeze(-1)

            out = torch.cat((out, next_step), dim=2)

            #derivatie_sv = torch.cat((derivatie_sv, pred), dim=2)


        for t in range(self.ws, one_full_traj.size(dim=2) - self.ws):

            seq = torch.cat((one_full_traj[:, 0:1, t : t + self.ws], out[:, :, t - self.ws : t]), dim=1)
            
            y1 = self.tcn(seq)
            pred = self.linear(y1[:, :, -1])

            next_step = out[:, :, t-1] + pred
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