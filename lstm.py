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

def train(input_data, model, weight_decay, future_decay):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)

    model.train()
    total_loss = []

    iterator = iter(input_data)

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x
        
        try:
            a,b = next(iterator)
            inp2 , label2 = next(iterator)
            inp2 , label2 = inp2.to(device) , label2.to(device)
            inp3 , label3 = next(iterator)
            inp3 , label3 = inp3.to(device) , label3.to(device)
            inp4 , label4 = next(iterator)
            inp4 , label4 = inp4.to(device) , label4.to(device)

        except StopIteration:
            break

        inp=inp.to(device)
        label=label.to(device)
        batch_loss = 0

        # Predict one timestep :
        output, _ = model(inp)
        out = inp[:, :, 1:] + output

        # Predict next 2 timesteps:
        #inp2 , label2 = next(iterator)
        next_inp = inp.clone()
        tmp = output.clone()
        next_inp[:, :, 1:] += tmp

        output2, _ = model(next_inp)
        out2 = inp2[:, :, 1:] + output2

        #inp3 , label3 = next(iterator)
        next_inp2 = next_inp.clone()
        tmp = output2.clone()
        next_inp2[:, :, 1:] += tmp

        output3, _ = model(next_inp2)
        out3 = inp3[:, :, 1:] + output3

        #inp4 , label4 = next(iterator)
        next_inp3 = next_inp2.clone()
        tmp = output3.clone()
        next_inp3[:, :, 1:] += tmp

        output4, _ = model(next_inp3)
        out4 = inp4[:, :, 1:] + output4
        
        optimizer.zero_grad(set_to_none=True)

        future_loss = future_decay * (loss_fn(out2[:,-1,:], label2[:, 1:]) +  loss_fn(out3[:,-1,:], label3[:, 1:]) +  loss_fn(out4[:,-1,:], label4[:, 1:]))
        if future_decay>0:
         future_loss.backward(retain_graph=True)

        # future_loss1 = future_decay * loss_fn(out2[:,-1,:], label2[:, 1:])
        # future_loss2 = future_decay *  loss_fn(out3[:,-1,:], label3[:, 1:])
        # future_loss3 = future_decay * loss_fn(out4[:,-1,:], label4[:, 1:])

        # if future_decay>0:
        #  future_loss1.backward(retain_graph=True)
        #  future_loss2.backward(retain_graph=True)
        #  future_loss3.backward(retain_graph=True)

        loss = loss_fn(out[:,-1,:], label[:, 1:])
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
    total_loss = 0

    for i, x in enumerate(test_data):
        x=x.to(device)
        if i > 20:
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

            total_loss += loss_fn(pred[:, 1:], x[:, 1:]).detach().cpu().numpy()

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
            
    return np.mean(test_loss), np.mean(test_loss_deriv), np.mean(total_loss)

def main():

    parameter_sets  = [


                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, weight_decay,  percentage_of_data     future_decay      batch_size
                        [16,           128 ,     3,    1000,        150,           0,           0,               0.8,                   0.3 ,             128],  

                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, weight_decay,  percentage_of_data     future_decay      batch_size
                        [16,           128 ,     3,    1000,        150,           0,           0,               0.8,                   0.8 ,             128], 

                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, weight_decay,  percentage_of_data     future_decay      batch_size
                        [16,           128 ,     3,    1000,        150,           0,           1e-5,               0.8,                 0.3 ,          128], 

                        #window_size, h_size, l_num, epochs, slice_of_data, part_of_data, weight_decay,  percentage_of_data     future_decay      batch_size
                        [16,           128 ,     3,    1000,        150,           0,          1e-5,               0.8,                  0.8 ,           128] 
                      ]


    for k,set in enumerate(parameter_sets):
        window_size, h_size, l_num, epochs, slice_of_data, part_of_data, weight_decay,  percentage_of_data, future_decay, batch_size = set
        
        # Configure logging
        log_file = 'training.log'
        filemode = 'a' if os.path.exists(log_file) else 'w'
        logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


        # Generate input data
        input_data = get_data(path = "save_data_test3.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 200, 
                                drop_half_timesteps = True,
                                normalise_s_w=True,
                                rescale_p=False,
                                num_inits=part_of_data)


        #Split data into train and test sets

        num_of_inits_train = int(len(input_data)*percentage_of_data)
        train_inits = np.random.randint(0,len(input_data), num_of_inits_train)
        train_inits = np.unique(train_inits)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])

        # make sure we really get the specified percentage of training data..
        if percentage_of_data < 0.99: 
                while len(train_inits) < num_of_inits_train * percentage_of_data:
                    i = np.random.randint(0,len(test_inits),1)[0]
                    train_inits = np.append(train_inits,test_inits[i])
                    test_inits = np.delete(test_inits, i)


        train_data = input_data[train_inits,:,:]
        test_data = input_data[test_inits,:,:]

        data_set  = CustomDataset(train_data, window_size=window_size)
        train_dataloader = DataLoader(data_set, batch_size=batch_size, pin_memory=True, drop_last=True)
        
        # Initialize the LSTM model
        model = LSTMmodel(input_size=3, hidden_size=h_size, out_size=2, layers=l_num).to(device)

        trained=False
        if trained:
            path = f"Ventil_trained_NNs\lstm_ws{window_size}.pth"
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))    
        
        #Train
        epochs=1
        losses = []
        average_traj_err_train = []
        average_traj_err_test = []

        for e in tqdm(range(epochs)):
            loss_epoch = train(train_dataloader, model, weight_decay, future_decay)

            losses.append(loss_epoch)
            #if e % 10 == 0:
            #    print(f"Epoch {e}: Loss: {loss_epoch}")

            if e%25 == 0:
                _,_, err_train = test(train_data, model, steps=input_data.size(dim=1), ws=window_size, plot_opt=False)
                _,_, err_test = test(test_data, model, steps=input_data.size(dim=1), ws=window_size, plot_opt=False)
                average_traj_err_train.append(err_train)
                average_traj_err_test.append(err_test)

        # Plot losses
        #plt.plot(average_traj_err_train[1:], label="inits from training data")
        #plt.plot(average_traj_err_test[1:], label="inits from testing data")
        #plt.title("Full trajectory prediction errors from initial value")
        #plt.legend()
        #plt.plot(losses[1:])
        #plt.show()

        # Save trained model
        k=np.random.randint(0,500,1)[0]
        path = f"Ventil_trained_NNs\lstm_ws{window_size}hs{h_size}layer{l_num}_nummer{k}_decay{future_decay}.pth"
        torch.save(model.state_dict(), path)
        print(f"Run finished, file saved as: \n {path}")


        # Log parameters
        logging.info(f"Epochs: {epochs}, Window Size: {window_size}")
        logging.info(f"hyperparams: h_size {h_size}, l_num {l_num}, epochs {epochs}, \n slice_of_data {slice_of_data},part_of_data {part_of_data}")
        logging.info(f"percentage_of_data {percentage_of_data}, weight_decay {weight_decay}, future_decay {future_decay}, batchsize {batch_size}")
        logging.info(f"final loss {losses[-1]}")
        logging.info(f"average training error every 40 epochs {average_traj_err_train}")
        logging.info(f"average test error every 40 epochs {average_traj_err_test}")

        logging.info(f"Final error over whole traj (average over some inits) {average_traj_err_train[-1]}")
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
