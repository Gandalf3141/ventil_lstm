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
import logging
import os
import cProfile
import pstats
from dataloader import *
from test_function_exp_NEW import *
from get_data import *
from NN_classes_NEW import *

#Define the LSTM model class
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)

def train_lstm_no_or_derivative(traindataloader, model, learning_rate=0.001):
   
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(traindataloader):  # inp = (u, x) label = x

        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output, _ = model.simple_forward(inp)
        out = inp[:,-1:, 1:] + output[:, -1:, :]

        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out[:,-1,:], label[:, 1:])

        loss.backward()
        optimizer.step()

        total_loss.append(loss.detach().cpu().numpy())

   # return the average error of the next step prediction
    return np.mean(total_loss)

def train_mlp_no_or_derivative(traindataloader, model, learning_rate=0.001):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    model.train()
    total_loss = []
  
    for k, (x, x_last, y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
        x_last = x_last.to(device)
        x_last = x_last.squeeze()
        
        output = model.simple_forward(x)
        pred = x_last[:,1:] + output[:, :]

        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(pred, y[:,1:])
        loss.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)

def train_tcn_no_or_derivative(traindataloader, model, learning_rate=0.001):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    model.train()
    total_loss = []
  
    for k, (x,y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)

        x = x.transpose(1,2)
        #y = y.transpose(1,2)

        output = model.simple_forward(x)
        out = x[:, 1:, -1:] + output[:, -1:].unsqueeze(-1)
  
        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(out[:,:,-1], y[:, 1:])
        loss.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)


def main():

    # test settings
    test_n = 2
    epochs = 20
    part_of_data = 10
    test_every_epochs = 5
    
    # Experiment settings
    # test_n = 50
    # epochs = 500
    # part_of_data = 100
    # test_every_epochs = 100

    batch_size_no_or = 500

    params_lstm =   {
                           "window_size" : 16,
                           "h_size" : 8,
                           "l_num" : 3,
                           "learning_rate" : 0.008,
                           "batch_size" : batch_size_no_or,
                    }

    params_mlp =    {
                           "window_size" : 20,
                           "h_size" : 24,
                           "l_num" : 3,
                           "learning_rate" : 0.001,
                           "batch_size" : batch_size_no_or,
                           "act_fn" : "relu",
                           "nonlin_at_out" : None #None if no nonlinearity at the end
                    }

    params_tcn =    {
                        "window_size" : 30,
                        "learning_rate" : 0.0005,
                        "batch_size" : batch_size_no_or,
                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "dropout" : 0
                    }
    
    parameter_configs  = [params_lstm, params_mlp, params_tcn] 


    for k, d in enumerate(parameter_configs):
        
        d["experiment_number"] = k
        d["epochs"] = epochs 
        d["input_channels"] = 3
        d["output"] = 2
        d["part_of_data"] = part_of_data
        d["percentage_of_data"] = 0.8
        d["drop_half_timesteps"] = True
        d["cut_off_timesteps"] = 0

    # Configure logging
    log_file = 'training_no_OR_derivative.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the LSTM model
    model_lstm = OR_LSTM(input_size=3, hidden_size=params_lstm["h_size"], out_size=2, layers=params_lstm["l_num"], window_size=params_lstm["window_size"]).to(device)
    
    # Initialize the MLP model
    model_mlp = OR_MLP(input_size=3*params_mlp["window_size"], hidden_size = params_mlp["h_size"], l_num=params_mlp["l_num"],
                    output_size=2, act_fn = params_mlp["act_fn"], act_at_end = params_mlp["nonlin_at_out"], timesteps=params_mlp["window_size"]).to(device)
    
    # Initialize the TCN model
    input_channels = params_tcn["input_channels"]
    output = params_tcn["output"]
    num_channels = [params_tcn["n_hidden"]] * params_tcn["levels"]
    kernel_size = params_tcn["kernel_size"]
    dropout = params_tcn["dropout"]
    model_tcn = OR_TCN(input_channels, output, num_channels, kernel_size=kernel_size, dropout=dropout, windowsize=params_tcn["window_size"]).to(device)

    # Generate input data (the data is normalized and some timesteps are cut off)
    input_data1, PSW_max = get_data(path = "data/save_data_test_revised.csv", 
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = params_tcn["drop_half_timesteps"],
                            normalise_s_w="minmax",
                            rescale_p=False,
                            num_inits=params_tcn["part_of_data"])
    
    input_data2, PSW_max = get_data(path = "data/save_data_test5.csv", 
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = params_tcn["drop_half_timesteps"],
                            normalise_s_w="minmax",
                            rescale_p=False,
                            num_inits=params_tcn["part_of_data"])
    
    input_data3, PSW_max = get_data(path = "data/Testruns_from_trajectory_generator_t2_t6_revised.csv", 
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = params_tcn["drop_half_timesteps"],
                            normalise_s_w="minmax",
                            rescale_p=False,
                            num_inits=params_tcn["part_of_data"])     

    test_data, PSW_max = get_data(path="data/save_data_test_5xlonger_dyndyn.csv",
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = True,
                            normalise_s_w="minmax",
                            rescale_p=False,
                            num_inits=0) 
    
    input_data = torch.cat((input_data1, input_data2, input_data3))
    print(input_data.size())

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*params_tcn["percentage_of_data"])
    train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
    np.random.shuffle(train_inits)
    np.random.shuffle(test_inits)
    train_data = input_data[train_inits,:input_data.size(dim=1)-params_tcn["cut_off_timesteps"],:]
    #test_data = input_data[test_inits,:,:]

    # dataloader for batching during training
    train_set_lstm = CustomDataset(train_data, window_size=params_lstm["window_size"])
    train_loader_lstm = DataLoader(train_set_lstm, batch_size=params_lstm["batch_size"], pin_memory=True)
    train_set_mlp = CustomDataset_mlp(train_data, window_size=params_mlp["window_size"])
    train_loader_mlp = DataLoader(train_set_mlp, batch_size=params_mlp["batch_size"], pin_memory=True)
    train_set_tcn = CustomDataset(train_data, window_size=params_tcn["window_size"])
    train_loader_tcn = DataLoader(train_set_tcn, batch_size=params_tcn["batch_size"], pin_memory=True)

    average_traj_err_train_lstm = []
    average_traj_err_train_mlp = []
    average_traj_err_train_tcn = []
    epochs = []
    average_traj_err_test = []

    #Training loop
    for e in tqdm(range(params_tcn["epochs"])):
        
        #train_lstm_no_or_derivative(train_loader_lstm, model_lstm, learning_rate= params_lstm["learning_rate"])
        #train_mlp_no_or_derivative(train_loader_mlp, model_mlp, learning_rate= params_mlp["learning_rate"])
        train_tcn_no_or_derivative(train_loader_tcn, model_tcn, learning_rate= params_tcn["learning_rate"])

        # Every few epochs get the error MSE of the true data
        # compared to the network prediction starting from some initial conditions
        if (e+1)%test_every_epochs == 0:
            _, err_train_lstm = test(test_data.to(device), model_lstm, model_type = "lstm_derivative", window_size=params_lstm["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)
            _, err_train_mlp = test(test_data.to(device), model_mlp, model_type = "mlp_derivative", window_size=params_mlp["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)
            _, err_train_tcn = test(test_data.to(device), model_tcn, model_type = "tcn_derivative", window_size=params_tcn["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)

            average_traj_err_train_lstm.append(err_train_lstm)
            average_traj_err_train_mlp.append(err_train_mlp)
            average_traj_err_train_tcn.append(err_train_tcn)
            epochs.append(e+1)
            
            print(f"Average error over full trajectories: training data LSTM: {err_train_lstm}")
            print(f"Average error over full trajectories: training data MLP: {err_train_mlp}")
            print(f"Average error over full trajectories: training data TCN: {err_train_tcn}")
            

    # Save trained model
    path_lstm = f'Trained_NNs_exp/LSTM_derivative_exp{params_lstm["experiment_number"]}.pth'
    path_mlp = f'Trained_NNs_exp/MLP_derivative_exp{params_mlp["experiment_number"]}.pth'
    path_tcn = f'Trained_NNs_exp/TCN_derivative_exp{params_tcn["experiment_number"]}.pth'

    torch.save(model_lstm.state_dict(), path_lstm)
    torch.save(model_mlp.state_dict(), path_mlp)
    torch.save(model_tcn.state_dict(), path_tcn)

    print(f"Run finished!")
    # Log parameters

    logging.info(f"hyperparams lstm: {params_lstm}")
    logging.info(f"hyperparams tcn: {params_mlp}")
    logging.info(f"hyperparams tcn: {params_tcn}")
    logging.info(f"Epochs {epochs}")
    logging.info(f"LSTM_derivative {average_traj_err_train_lstm}")
    logging.info(f"MLP_derivative {average_traj_err_train_mlp}")
    logging.info(f"TCN_derivative {average_traj_err_train_tcn}")   
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

if __name__ == "__main__":
    main()
