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
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

print(device)


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

        last = inp[-1:,:]

        inp = torch.cat((inp[:,0], inp[:,1], inp[:,2]))
        
        return inp, last, label

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
        
        # Use nn.Sequential to put together the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.network(x)         
  
def test(test_data, model, steps=600, ws=10, plot_opt=False, n = 5, test_inits=1, rand=True, PSW_max = 0):
 
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0
    total_loss = 0
   
    if rand:
     np.random.seed(1234)

 
    ids = np.random.choice(test_inits, min([n, test_inits]), replace=False)
    ids = np.unique(ids)
 
 
    for i, x in enumerate(test_data):

        x=x.to(device)
        
        if i not in ids:
            continue
 
        with torch.inference_mode():
 
            pred = torch.zeros((steps, 3), device=device)
 
            if ws > 1:
                pred[0:ws, :] = x[0:ws, :]
                pred[:, 0] = x[ :, 0]
 
            else:
                pred[0, :] = x[0, :]
                pred[:, 0] = x[:, 0]

            inp = torch.cat((x[:ws,0], x[:ws,1], x[:ws,2]))

            for t in range(1,steps - ws + 1 ): 

                out = model(inp)
                pred[ws+(t-1):ws+t,1:] =  pred[ws+(t-2):ws+(t-1):,1:] + out
                new_p = pred[t:t+ws,0]
                new_s = pred[t:t+ws,1]
                new_v = pred[t:t+ws,2]
                
                inp = torch.cat((new_p, new_s, new_v))

            test_loss += loss_fn(pred[ws:, 1], x[ws:, 1]).detach().cpu().numpy()
            test_loss_deriv += loss_fn(pred[ws:, 2], x[ws:, 2]).detach().cpu().numpy()
            total_loss += loss_fn(pred[ws:, 1:], x[ws:, 1:]).detach().cpu().numpy()

            #scale back:    
            if PSW_max != 0:
                pred[:,0] = pred[:,0]*PSW_max[0]
                pred[:,1] = pred[:,1]*PSW_max[1]
                pred[:,2] = (pred[:,2]*2 - 1) * PSW_max[2]
                x[:,0] = x[:,0]*PSW_max[0]
                x[:,1] = x[:,1]*PSW_max[1]
                x[:,2] = (x[:,2]*2 - 1) * PSW_max[2]

            if plot_opt:
                figure , axs = plt.subplots(1,3,figsize=(20,9))
           
                axs[0].plot(pred.detach().cpu().numpy()[:, 1], color="red", label="pred")
                axs[0].plot(x.detach().cpu().numpy()[:, 1], color="blue", label="true", linestyle="dashed")
                axs[0].set_title("position")
                axs[0].set_ylabel("[m]")
                axs[0].set_xlabel("time")
                axs[0].grid()
                axs[0].legend()
 
                axs[1].plot(pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
                axs[1].plot(x.detach().cpu().numpy()[:, 2], color="blue", label="true", linestyle="dashed")
                axs[1].set_title("speed")
                axs[1].set_ylabel("[m/s]")
                axs[1].set_xlabel("time")
                axs[1].grid()
                axs[1].legend()
 
                axs[2].plot(x.detach().cpu().numpy()[:,0], label="pressure")
                axs[2].set_title("pressure")
                axs[2].set_ylabel("[Pascal]")
                axs[2].set_xlabel("time")
                axs[2].grid()
                axs[2].legend()
 
                plt.grid(True)
                plt.legend()
                plt.show()
           
    return np.mean(test_loss), np.mean(test_loss_deriv), np.mean(total_loss)

def train(loader, model, weight_decay, learning_rate=0.001, ws=0, batch_size=1):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
 
    model.train()
    total_loss = []
  
    for k, (x, x_last, y) in enumerate(loader):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
        x_last = x_last.to(device)
        
        output = model(x)
        pred = x_last[0,:,1:] + output

        # reset the gradient

        if k % batch_size == 0:

            optimizer.zero_grad(set_to_none=True)
            
            # calculate the error
            loss = loss_fn(pred, y[:,1:])
            loss.backward()
            optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)

def main():
                        
    parameter_configs  = [
                        {
                           "experiment_number" : 2,
                           "window_size" : 5,
                           "h_size" : 16,
                           "l_num" : 3,
                           "epochs" : 2000,
                           "learning_rate" : 0.001,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.5,
                           "batch_size" : 600,
                           "cut_off_timesteps" : 300,
                           "drop_half_timesteps": True,
                           "act_fn" : "relu"
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
        model = MLP(input_size=3*params["window_size"], hidden_size = params["h_size"], l_num=params["l_num"], output_size=2, act_fn = params["act_fn"]).to(device)

        # Generate input data (the data is normalized and some timesteps are cut off)
        input_data, PSW_max = get_data(path = "save_data_test4.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])
        
        input_data2, PSW_max = get_data(path = "save_data_test5.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])
        
        input_data3, PSW_max = get_data(path = "Testruns_from_trajectory_generator_200.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])       


        input_data = torch.cat((input_data, input_data2, input_data3))


        print(input_data.size())

        #Split data into train and test sets
        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])

        train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
        test_data = input_data[test_inits,:,:]

        # dataloader for batching during training
        train_set = CustomDataset(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set)#, batch_size=params["batch_size"], pin_memory=True)

        losses = []
        average_traj_err_train = []
        average_traj_err_test = []

        for e in tqdm(range(params["epochs"])):
            
            train(train_loader, model, params["weight_decay"], learning_rate= params["learning_rate"], ws=params["window_size"], batch_size=params["batch_size"])

            # Every few epochs get the error MSE of the true data
            # compared to the network prediction starting from some initial conditions
            if (e+1)%200 == 0:
                _,_, err_train = test(train_data, model, steps=train_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 20)
               # _,_, err_test = test(test_data, model, steps=test_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 40)
                average_traj_err_train.append(err_train)
              #  average_traj_err_test.append(err_test)

                print(f"Average error over full trajectories: training data : {err_train}")
                #print(f"Average error over full trajectories: testing data : {err_test}")

        _,_, err_train = test(train_data, model, steps=train_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 100)
        #_,_, err_test = test(test_data, model, steps=test_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 100)
        print(f"TRAINING FINISHED: Average error over full trajectories: training data : {err_train}")
       # print(f"TRAINING FINISHED: Average error over full trajectories: testing data : {err_test}")
        
        # Save trained model
        path = f'Ventil_trained_NNs\MLP{params["experiment_number"]}.pth'
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
