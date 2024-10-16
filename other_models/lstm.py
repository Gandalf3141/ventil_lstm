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
from test_function import * 
from NN_classes import *

# Define the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#works:
def train(input_data, model, weight_decay, future_decay, learning_rate=0.001, ws=0, future=1):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x
        #print(k, f"timesteps {k} : {k+4} mit label bis {k+6}")
        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output, _ = model(inp)
        out = inp[:, :, 1:] + output

        # print("inp", inp, inp.size())
        # print("label", label, label.size())
        # print("out", out, out.size())
        
        # #1. extra step-------------------------
        # if future>1:
        #     new_combined_inp = torch.cat((label[:, 0, 0:1], out[:,-1,:]), dim=1)
        #     new_combined_inp = new_combined_inp.view(inp.size(dim=0),1,3)

        #     #print("new_combined_inp", new_combined_inp, new_combined_inp.size())

        #     inp2 = torch.cat((inp[: , 1:ws,:], new_combined_inp), dim =1)        
        #     #print("inp2" , inp2, inp2.size())

        #     output2, _ = model(inp2)
        #     out2 = inp2[:, :, 1:] + output2

        #     #print("out2", out2, out2.size())

        # #2. extra step-------------------------
        # if future > 2:
        #     #new_combined_inp2 = torch.cat((label[:, 1, 0:1], out2[:,-1,:].clone()), dim=1)
        #     new_combined_inp2 = torch.cat((label[:, 1, 0:1], out2[:,-1,:]), dim=1)
        #     new_combined_inp2 = new_combined_inp2.view(inp2.size(dim=0),1,3)

        #     inp3 = torch.cat((inp2[: , 1:ws,:], new_combined_inp2), dim =1)        

        #     output3, _ = model(inp3)
        #     out3 = inp3[:, :, 1:] + output3
        
        # #3. extra step-------------------------
        # if future > 3:
        #     new_combined_inp3 = torch.cat((label[:, 1, 0:1], out3[:,-1,:].clone()), dim=1)
        #     new_combined_inp3 = new_combined_inp3.view(inp2.size(dim=0),1,3)

        #     inp4 = torch.cat((inp3[: , 1:ws,:], new_combined_inp3), dim =1)        

        #     output4, _ = model(inp4)
        #     out4 = inp4[:, :, 1:] + output4

        # # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error
        if future<2:
            loss = loss_fn(out[:,-1,:], label[:, 1:])
        else:   
            loss = loss_fn(out[:,-1,:], label[:, 0, 1:])

        # #backpropagation
        # if future>1:
        #     loss2 = future_decay * loss_fn(out2[:,-1,:], label[:, 1, 1:])
        #     loss += loss2
        # if future>2:
        #     loss3 = future_decay * loss_fn(out3[:,-1,:], label[:, 2, 1:])
        #     loss += loss3
        # if future>3:
        #     loss4 = future_decay * loss_fn(out4[:,-1,:], label[:, 3, 1:])
        #     loss += loss4

        loss.backward(retain_graph=False)
        optimizer.step()


        total_loss.append(loss.detach().cpu().numpy())

   # return the average error of the next step prediction
    return np.mean(total_loss)

def main():
                                #{'lr': 0.00020110091342376562, 'ws': 4, 'bs': 256, 'hs': 6}#
    parameter_configs  = [
                        {

                           "experiment_number" : 4,
                           "window_size" : 4,
                           "h_size" : 5,
                           "l_num" : 1,
                           "epochs" : 500,
                           "learning_rate" : 0.0008,
                           "part_of_data" : 0, 
                           "weight_decay" : 1e-5,
                           "percentage_of_data" : 0.8,
                           "future_decay"  : 0.1,
                           "batch_size" : 2000,
                           "future" : 4,
                           "drop_half_timesteps" : True
                           
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
        model = LSTMmodel(input_size=3, hidden_size=params["h_size"], out_size=2, layers=params["l_num"]).to(device)

        # Generate input data (the data is normalized and some timesteps are cut off)
        input_data, PSW_max = get_data(path = "save_data_test4.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        cut_off_timesteps = 800
        #Split data into train and test sets

        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
        np.random.shuffle(train_inits)
        np.random.shuffle(test_inits)
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
                _,_, err_train = test(train_data, model, model_type = "lstm", window_size=params["window_size"], display_plots=False, num_of_inits = 20, set_rand_seed=True, physics_rescaling = PSW_max)
               # _,_, err_test = test(test_data, model, steps=test_data.size(dim=1), ws=params["window_size"], plot_opt=False, n = 40)
                average_traj_err_train.append(err_train)
              #  average_traj_err_test.append(err_test)

                print(f"Average error over full trajectories: training data : {err_train}")
                #print(f"Average error over full trajectories: testing data : {err_test}")

        _,_, err_train = test(train_data, model, model_type = "lstm", window_size=params["window_size"], display_plots=False, num_of_inits = 100, set_rand_seed=True, physics_rescaling = PSW_max)
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
