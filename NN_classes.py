import torch
from torch import nn

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

                #wäre richtig!!!    
                seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:1], out[:, t - self.ws : t , :]), dim=2)
                #war falsch
                #seq = torch.cat((out[:, t - self.ws : t , :],one_full_traj[:, t : t + self.ws, 0:1]), dim=2)
                
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
    
    def __init__(self, input_size=3, hidden_size = 6, l_num=1, output_size=2, act_fn="tanh"):
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
       # layers.append(nn.Sigmoid())
        
        # Use nn.Sequential to put together the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.network(x)         


