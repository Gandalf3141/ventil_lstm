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

 #Define the LSTM model class

# Use the GPU if available
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device="cpu"
print(device)

class LSTMmodel(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, future=1):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.future = future
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):
    
        lstm_out, hidden = self.lstm(seq)
        #lstm_out = self.act(lstm_out)
        pred = self.linear(lstm_out)

        if self.future==1:
            return pred, hidden
        
        out = []
        if self.future > 1:
            for t in range(self.future):
                lstm_out, hidden = self.lstm(seq, hidden)
                #lstm_out = self.act(lstm_out)
                pred = self.linear(lstm_out)
                out.append(pred) 
        return out, hidden         


#works:
def train(input_data, model, weight_decay, future_decay, learning_rate=0.001, ws=0, future=1):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x

        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output, _ = model(inp)

        out = []
        for i in range(future):
            out.append(inp[:, :, 1:] + output[i])

        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out[0][:,-1,:], label[:, 0, 1:])

        for t in range(1, future):
            loss += (future_decay**t) * loss_fn(out[t][:,-1,:], label[:, t, 1:])

        loss.backward(retain_graph=True)
        optimizer.step()


        total_loss.append(loss.detach().cpu().numpy())

   # return the average error of the next step prediction
    return np.mean(total_loss)



def test(test_data, model, steps=600, ws=10, plot_opt=False, n = 5, rand=True):

    #test_data = test_dataloader.get_all_data() 
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0
    total_loss = 0
    
    if rand:
     np.random.seed(123)
 
    ids = np.random.choice(test_data.size(dim=0), min([n, test_data.size(dim=0)]), replace=False)
    ids = np.unique(ids)

    #print(ids)

    for i, x in enumerate(test_data):
        x=x.to(device)
        if i not in ids:
            continue

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

                plt.grid(True)
                plt.legend()
                plt.show()
            
    return np.mean(test_loss), np.mean(test_loss_deriv), np.mean(total_loss)


def main():
                                #{'lr': 0.00020110091342376562, 'ws': 4, 'bs': 256, 'hs': 6}#
    parameter_configs  = [
                        {
                           "experiment_number" : 4,
                           "window_size" : 4,
                           "h_size" : 5,
                           "l_num" : 1,
                           "epochs" : 100,
                           "learning_rate" : 0.0002,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 64,
                           "future" : 10
                        },
                                                {
                           "experiment_number" : 12,
                           "window_size" : 4,
                           "h_size" : 5,
                           "l_num" : 1,
                           "epochs" : 100,
                           "learning_rate" : 0.0005,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 64,
                           "future" : 1
                        },
                                                {
                           "experiment_number" : 523,
                           "window_size" : 4,
                           "h_size" : 10,
                           "l_num" : 3,
                           "epochs" : 100,
                           "learning_rate" : 0.0002,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 512,
                           "future" : 5
                        },
                                                {
                           "experiment_number" : 55,
                           "window_size" : 4,
                           "h_size" : 10,
                           "l_num" : 3,
                           "epochs" : 80,
                           "learning_rate" : 0.0002,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 5012,
                           "future" : 5
                        },
                        {
                           "experiment_number" : 67456,
                           "window_size" : 4,
                           "h_size" : 5,
                           "l_num" : 1,
                           "epochs" : 200,
                           "learning_rate" : 0.0005,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 8000,
                           "future" : 4
                        }

                      ]

    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k

    for k, params in enumerate(parameter_configs):

        # Configure logging
        log_file = 'training.log'
        filemode = 'a' if os.path.exists(log_file) else 'w'
        logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize the LSTM model
        model = LSTMmodel(input_size=3, hidden_size=params["h_size"], out_size=2, layers=params["l_num"], future=params["future"]).to(device)

        # Generate input data (the data is normalized and some timesteps are cut off)
        input_data = get_data(path = "save_data_test4.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = False,
                                normalise_s_w=True,
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        cut_off_timesteps = 600
        #Split data into train and test sets

        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])

        train_data = input_data[train_inits,:input_data.size(dim=1)-cut_off_timesteps,:]
        test_data = input_data[test_inits,:,:]

        data_set  = CustomDataset(train_data, window_size=params["window_size"], future=params["future"])
        train_dataloader = DataLoader(data_set, batch_size=params["batch_size"], pin_memory=True, drop_last=True)

        average_traj_err_train = []
        average_traj_err_test = []

        for e in tqdm(range(params["epochs"])):
            
            train(train_dataloader, model, params["weight_decay"], params["future_decay"],
                  learning_rate=params["learning_rate"], ws=params["window_size"], future=params["future"])

            # Every few epochs get the error MSE of the true data
            # compared to the network prediction starting from some initial conditions
            if (e+1)%10 == 0:
                model.future=1
                _,_, err_train = test(train_data, model, steps=train_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 40)
                model.future=params["future"]
                if err_train < 2:
                    print("stopped early!")
                    break
               # _,_, err_test = test(test_data, model, steps=test_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 40)
                average_traj_err_train.append(err_train)
              #  average_traj_err_test.append(err_test)

                print(f"Average error over full trajectories: training data : {err_train}")
                #print(f"Average error over full trajectories: testing data : {err_test}")

        model.future=1
        _,_, err_train = test(train_data, model, steps=train_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 100)
        model.future=params["future"]
        #_,_, err_test = test(test_data, model, steps=test_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 100)
        print(f"TRAINING FINISHED: Average error over full trajectories: training data : {err_train}")
       # print(f"TRAINING FINISHED: Average error over full trajectories: testing data : {err_test}")
        
        # Save trained model
        path = f'Ventil_trained_NNs\lstm_ws{params["experiment_number"]}.pth'
        torch.save(model.state_dict(), path)
        print(f"Run finished, file saved as: \n {path}")

        # Log parameters
        logging.info(f"hyperparams: {params}")
        logging.info(f"Final train error over whole traj (average over some inits) {average_traj_err_train}")
        logging.info(f"Final test error over whole traj (average over some inits) {average_traj_err_test}")
        logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
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
