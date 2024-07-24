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
from test_function_exp import *
from get_data import *
from NN_classes import *
from nextstep_NN_classes import *

#Define the LSTM model class
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)


def train_tcn(input_data, model, learning_rate=0.001):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    model.train()
    total_loss = []
  
    for k, (x,y) in enumerate(input_data):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)

        x = x.transpose(1,2)
        y = y.transpose(1,2)

        out = model(x)
  
        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)


def train_tcn_nextstep(input_data, model, learning_rate=0.001):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    model.train()
    total_loss = []
  
    for k, (x,y) in enumerate(input_data):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)

        x = x.transpose(1,2)
        y = y.transpose(1,2)

        out = model(x)
  
        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)


def main():

    # test settings
    test_n = 1
    epochs = 20
    part_of_data = 20
    test_every_epochs = 4
    
    # Experiment settings
    # test_n = 100
    # epochs = 2000
    # part_of_data = 0
    # test_every_epochs = 200

    params_tcn =    {
                        "window_size" : 30,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "dropout" : 0
                    }
    
    parameter_configs  = [params_tcn] 

    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k
        d["epochs"] = epochs
        d["input_channels"] = 3
        d["output"] = 2
        d["part_of_data"] = part_of_data
        d["percentage_of_data"] = 0.99
        d["drop_half_timesteps"] = True
        d["cut_off_timesteps"] = 100

    # Configure logging
    log_file = 'training_OR_nets.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    # Initialize the TCN model
    input_channels = params_tcn["input_channels"]
    output = params_tcn["output"]
    num_channels = [params_tcn["n_hidden"]] * params_tcn["levels"]
    kernel_size = params_tcn["kernel_size"]
    dropout = params_tcn["dropout"]
    model_tcn = OR_TCN(input_channels, output, num_channels, kernel_size=kernel_size, dropout=dropout, windowsize=params_tcn["window_size"]).to(device)
    model_tcn_or_nextstep = TCN_or_nextstep(input_channels, output, num_channels, kernel_size=kernel_size, dropout=dropout, windowsize=params_tcn["window_size"]).to(device)

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

    
    input_data = torch.cat((input_data1, input_data2, input_data3))
    print(input_data.size())

    #Split data into train and test sets

    #Use a mask to select trajectories where the initial position is smaller then 0.6
    ####
    mask = input_data[:, 0, 1] < 0.6

    train_data = input_data[mask]

    train_data = train_data[:,:train_data.size(dim=1)-params_tcn["cut_off_timesteps"], :]
    test_data = input_data[~mask]
    print("Size of training data:" , train_data.size())
    print("Size of testing data:" , test_data.size())

    # dataloader for batching during training
    train_set_tcn = custom_simple_dataset(train_data, window_size=params_tcn["window_size"])
    train_loader_tcn = DataLoader(train_set_tcn, batch_size=params_tcn["batch_size"], pin_memory=True)
    
    average_traj_err_test_tcn = []
    average_traj_err_test_tcn_nextstep = []
    average_traj_err_train_tcn = []
    average_traj_err_train_tcn_nextstep = []    
    epochs = []

    #Training loop
    for e in tqdm(range(params_tcn["epochs"])):
        

        train_tcn(train_loader_tcn, model_tcn, learning_rate= params_tcn["learning_rate"])
        train_tcn_nextstep(train_loader_tcn, model_tcn_or_nextstep, learning_rate= params_tcn["learning_rate"])


        # Every few epochs get the error MSE of the true data
        # compared to the network prediction starting from some initial conditions
        if (e+1)%test_every_epochs == 0:
            _,_, err_test_tcn = test(test_data.to(device), model_tcn, model_type = "or_tcn", window_size=params_tcn["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)
            _,_, err_test_tcn_nextstep = test(test_data.to(device), model_tcn_or_nextstep, model_type = "tcn_or_nextstep", window_size=params_tcn["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)

            _,_, err_train_tcn = test(train_data.to(device), model_tcn, model_type = "or_tcn", window_size=params_tcn["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)
            _,_, err_train_tcn_nextstep = test(train_data.to(device), model_tcn_or_nextstep, model_type = "tcn_or_nextstep", window_size=params_tcn["window_size"], display_plots=False, num_of_inits = test_n, set_rand_seed=True, physics_rescaling = PSW_max)

            average_traj_err_test_tcn.append(err_test_tcn)
            average_traj_err_test_tcn_nextstep.append(err_test_tcn_nextstep)
            average_traj_err_train_tcn.append(err_train_tcn)
            average_traj_err_train_tcn_nextstep.append(err_train_tcn_nextstep)
            epochs.append(e+1)

            print(f"Average error over full trajectories: test data TCN_OR: {err_test_tcn}")
            print(f"Average error over full trajectories: test data TCN_OR_nextstep: {err_test_tcn_nextstep}")
            print(f"Average error over full trajectories: train data TCN_OR: {err_train_tcn}")
            print(f"Average error over full trajectories: train data TCN_OR_nextstep: {err_train_tcn_nextstep}")    

    # Save trained model
    path_tcn = f'Ventil_trained_NNs/OR_TCN_sparsedata_exp{params_tcn["experiment_number"]}.pth'
    path_tcn_nextstep = f'Ventil_trained_NNs/TCN_or_nextstep_sparsedata_exp{params_tcn["experiment_number"]}.pth'

    torch.save(model_tcn.state_dict(), path_tcn)
    torch.save(model_tcn_or_nextstep.state_dict(), path_tcn_nextstep)

    print(f"Run finished!")

    # Log parameters
    logging.info(f"hyperparams tcn: {params_tcn}")
    logging.info(f"Epochs {epochs}")
    logging.info(f"TCN test {average_traj_err_test_tcn}")  
    logging.info(f"TCN_nextstep test {average_traj_err_test_tcn_nextstep}")   
    logging.info(f"TCN train {average_traj_err_train_tcn}")  
    logging.info(f"TCN_nextstep train {average_traj_err_train_tcn_nextstep}") 
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

if __name__ == "__main__":
    main()
