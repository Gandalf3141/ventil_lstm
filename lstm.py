# Importing necessary libraries
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
from icecream import ic
from tqdm import tqdm
from itertools import chain
from get_data import get_data
import logging
import os
import cProfile
import pstats
from dataloader import *

# Define the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

class LSTMmodel(nn.Module):
    """
    LSTM model class for derivative estimation.
    """

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

        # Define LSTM layer
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
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
        pred = self.linear(lstm_out)#.view(len(seq), -1))

        return pred, hidden


def slice_batch(batch, window_size=1):
    """
    Slice the input data into batches for training.

    Args:
    - batch: Input data batch
    - window_size: Size of the sliding window

    Returns:
    - List of sliced batches
    """
    l = []
    for i in range(len(batch) - window_size):
        l.append((batch[i:i+window_size, :], batch[i+1:i+window_size+1, 1:]))
    return l


def train(input_data, model):
    """
    Train the LSTM model using input data.

    Args:
    - input_data: Input data for training
    - model: LSTM model to be trained
    - ws: Window size
    - odestep: Option for using ODE steps
    - use_autograd: Option for using autograd

    Returns:
    - Mean loss over all batches
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())#, lr=0.001)

    model.train()
    total_loss = []

    iterator = iter(input_data)

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x

        a, b = next(iterator)
        
        inp=inp.to(device)
        label=label.to(device)
        batch_loss = 0

        #print("inp",inp.size())
        #print("label",label.size())
        output, _ = model(inp)
        #print("output", output.size())

        #reconsider this part :        
        #maybe | out = inp[-1, 1:] + output[-1] | works better
        #out = inp[:, 1:] + output
        #out = inp[:,-1, 1:] + output[:,-1,:]
        out = inp[:, :, 1:] + output

        next_inp = inp.clone()
        tmp = output.clone()
        next_inp[:, :, 1:] += tmp

        output2, _ = model(next_inp)
        out2 = inp[:, :, 1:] + output2
        #print("out",out.size())

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(out[:,-1,:], label[:, 1:])
        loss2 = loss_fn(out2[:,-1,:], b[:, 1:])
        loss2.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss.append(loss.detach().cpu().numpy())

    return np.mean(total_loss)


def test(test_data, model, steps=600, ws=10, plot_opt=False):

    #test_data = test_dataloader.get_all_data() 
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0

    for i, x in enumerate(test_data):
        x=x.to(device)
        if i > 2:
            break

        with torch.inference_mode():

            pred = torch.zeros((steps, 3), device=device)
            pred_next_step = torch.zeros((steps, 3), device=device)

            if ws > 1:
                pred[0:ws, :] = x[0:ws, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0:ws, :] = x[0:ws, :]
                pred_next_step[:, 0] = x[:, 0]
            else:
                pred[0, :] = x[0, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0, :] = x[0, :]
                pred_next_step[:, 0] = x[:, 0]

            for i in range(len(x) - ws):

                out, _ = model(pred[i:i+ws, :])
                pred[i+ws, 1:] = pred[i+ws-1, 1:] + out[-1, :]
                pred_next_step[i+ws, 1:] = x[i+ws-1, 1:] + out[-1, :]
            
            test_loss += loss_fn(pred[:, 1], x[:, 1]).detach().cpu().numpy()
            test_loss_deriv += loss_fn(pred[:, 2], x[:, 2]).detach().cpu().numpy()

            if plot_opt:
                figure , axs = plt.subplots(1,3,figsize=(16,9))
            
                axs[0].plot(pred.detach().cpu().numpy()[:, 1], color="red", label="pred")
                axs[0].plot(pred_next_step.detach().cpu().numpy()[:, 1], color="green", label="next step from data")
                axs[0].plot(x.detach().cpu().numpy()[:, 1], color="blue", label="true", linestyle="dashed")
                axs[0].set_title("position")
                axs[0].grid()
                axs[0].legend()

                axs[1].plot(pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
                axs[1].plot(pred_next_step.detach().cpu().numpy()[:, 2], color="green", label="next step from data")
                axs[1].plot(x.detach().cpu().numpy()[:, 2], color="blue", label="true", linestyle="dashed")
                axs[1].set_title("speed")
                axs[1].grid()
                axs[1].legend()

                axs[2].plot(x.detach().cpu().numpy()[:,0], label="pressure")
                axs[2].set_title("pressure")
                axs[2].grid()
                axs[2].legend()

                plt.grid()
                plt.legend()
                plt.show()
            
    return np.mean(test_loss), np.mean(test_loss_deriv)

def main():

    parameter_sets  = [
                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, part_of_old_data,  percentage_of_data
                        [1,           64 ,     3,     2,        150,           0,           0,               0.1], 

                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, part_of_old_data,  percentage_of_data
                        #[2,           128 ,     3,     100,        150,           0,           0,               0.7],  


                      ]

    for k,set in enumerate(parameter_sets):
        window_size, h_size, l_num, epochs, slice_of_data, part_of_data, part_of_old_data,  percentage_of_data = set
        


        log_file = 'training.log'
        filemode = 'a' if os.path.exists(log_file) else 'w'

        # Configure logging
        logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Define parameters
        # window_size =4 
        # h_size=5
        # l_num=1
        losses = []

        # Generate input data
        input_data = get_data(path = "save_data_test3.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 800, 
                                drop_half_timesteps = True,
                                normalise_s_w=True,
                                rescale_p=False,
                                num_inits=part_of_data)
        
        input_data_old = get_data(path = "save_data_test.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = True,
                                normalise_s_w=True,
                                rescale_p=False,
                                num_inits=part_of_old_data)
        if part_of_old_data>0:
         input_data=torch.cat((input_data, input_data_old))
         
        input_test = get_data(path = "save_data_validate.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = True,
                                normalise_s_w=True,
                                rescale_p=False,
                                num_inits=10)
        
        data  = CustomDataset(input_data, window_size=window_size)

       #Split data into train and test sets
        train_size = int(percentage_of_data * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=64,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
  
        # Take a slice of data for training (only slice_of_data many timesteps)
        #slice_of_data = 10

        # Initialize the LSTM model
        model = LSTMmodel(input_size=3, hidden_size=h_size, out_size=2, layers=l_num).to(device)

        trained=False
        if trained:
            path = f"Ventil_trained_NNs\lstm_ws{window_size}.pth"
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))

        
        #Train
        #epochs=1
        for e in tqdm(range(epochs)):
            loss_epoch = train(train_dataloader, model)

            losses.append(loss_epoch)
            if e % 5 == 0:

                print(f"Epoch {e}: Loss: {loss_epoch}")
                #print(test(input_data, model, steps=300, ws=window_size, plot_opt=False))
    
        # Plot losses
        #plt.plot(losses[1:])
        #plt.show()

        # Save trained model
        path = f"Ventil_trained_NNs\lstm_ws{window_size}hs{h_size}layer{l_num}_nummer{k}.pth"
        torch.save(model.state_dict(), path)

        #test the model
        #test(input_data2, model, steps=input_data.size(dim=1), ws=window_size, plot_opt=True)

        # Log parameters
        logging.info(f"Epochs: {epochs}, Window Size: {window_size}")
        logging.info(f"hyperparams: h_size, l_num, epochs, slice_of_data, part_of_data, part_of_old_data,  percentage_of_data")
        logging.info(f"hyperparams: {h_size,      l_num,     epochs,    slice_of_data,    part_of_data,    part_of_old_data,    percentage_of_data}")
        logging.info(f"final loss {losses[-1]}")
        logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        logging.info("\n")
        logging.info("\n")




if __name__ == "__main__":

    main()

    # profiler = cProfile.Profile()
    # profiler.enable()
    # main(window_size=16, h_size=8, l_num=2, epochs=100, slice_of_data=50)
    # profiler.disable()
    
    # stats = pstats.Stats(profiler)
    # # Sort the statistics by cumulative time
    # stats.sort_stats("cumulative")
    # # Print the top 10 functions with the highest cumulative time
    # stats.print_stats(10)

# chat gpt instructions
# Add short and precise comments to the following python code. Describe functions, classes, loops similar things. The code:
