# Importing necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from icecream import ic
from tqdm import tqdm
from get_data import *
from dataloader import *
from test_function import test
from NN_classes import *
import logging
import os 

from pytorch_tcn import TCN


# Use the GPU if available
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

def train(input_data, model, weight_decay, learning_rate=0.001):


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x

        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output = model(inp)
        out = output[:, -1:, :]

        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out[:,-1,:], label[:, 0, 1:])

        loss.backward(retain_graph=True)
        optimizer.step()


        total_loss.append(loss.detach().cpu().numpy())

   # return the average error of the next step prediction
    return np.mean(total_loss)


def main():
                
    parameter_configs  = [
                            
                                     {
                           "experiment_number" : 4,
                           "window_size" : 20,
                           "h_size" : 5,
                           "l_num" : 1,
                           "epochs" : 500,
                           "learning_rate" : 0.0005,
                           "part_of_data" : 0, 
                           "weight_decay" : 0,
                           "percentage_of_data" : 0.7,
                           "future_decay"  : 0.1,
                           "batch_size" : 50,
                           "future" : 4,
                           "drop_half_timesteps" : True,
                           "cut_off_timesteps" : 0
                        }


    ]

    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k

    for k, params in enumerate(parameter_configs):

        # Configure logging
        log_file = 'training.log'
        filemode = 'a' if os.path.exists(log_file) else 'w'
        logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the model

        model = TCN(

            3, # num_inputs: int,
            [4,4,4, 2],# num_channels: ArrayLike,
            3,         # kernel_size: int = 4,
            None,# dilations: Optional[ ArrayLike ] = None,
            8, # dilation_reset: Optional[ int ] = None,
            0.01,# dropout: float = 0.1
            True,# causal: bool = True,
            "weight_norm",# use_norm: str = 'weight_norm',
            'relu',# activation: str = 'relu',
            'xavier_uniform',# kernel_initializer: str = 'xavier_uniform',
            False,# use_skip_connections: bool = False,
            'NLC',# input_shape: str = 'NCL',
        ).to(device)


        # Generate input data (the data is normalized and some timesteps are cut off)
        input_data1, PSW_max = get_data(path = "data\save_data_test_revised.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])
        
        input_data2, PSW_max = get_data(path = "data\save_data_test5.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])
        
        input_data3, PSW_max = get_data(path = "data\Testruns_from_trajectory_generator_t2_t6_revised.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        input_data = torch.cat((input_data1, input_data2, input_data3))
        input_data =  input_data1
        #input_data = torch.cat((input_data1, input_data3))


        print(input_data.size())

        #Split data into train and test sets
        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        #num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
        np.random.shuffle(train_inits)
        np.random.shuffle(test_inits)

        train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
        test_data = input_data[test_inits,:,:]
        print(train_data.size())

        data_set  = CustomDataset(train_data, window_size=params["window_size"], future=params["future"])
        train_dataloader = DataLoader(data_set, batch_size=params["batch_size"], pin_memory=True, drop_last=True)

        losses = []
        average_traj_err_train = []
        average_traj_err_test = []

        for e in tqdm(range(params["epochs"])):
            
            loss_epoch = train(train_dataloader, model, params["weight_decay"], learning_rate=params["learning_rate"])
            losses.append(loss_epoch)

            # Every few epochs get the error MSE of the true data
            # compared to the network prediction starting from some initial conditions
            if (e+1)%25 == 0:

                _,_, err_train = test(train_data.to(device=device), model, model_type = "tcn", window_size=params["window_size"], display_plots=False, num_of_inits = 10, set_rand_seed=True, physics_rescaling = PSW_max)
                _,_, err_test = test(test_data.to(device=device), model, model_type = "tcn", window_size=params["window_size"], display_plots=False, num_of_inits = 10, set_rand_seed=True, physics_rescaling = PSW_max)
                average_traj_err_train.append(err_train)
                average_traj_err_test.append(err_test)
                print(f"Epoch: {e}, the average next step error was : loss_epoch")
                print(f"Average error over full trajectories: training data : {err_train}")
                print(f"Average error over full trajectories: testing data : {err_test}")  

        _, _, err_test_final = test(test_data.to(device=device), model, model_type = "tcn", window_size=params["window_size"], display_plots=False, num_of_inits = 100, set_rand_seed=True, physics_rescaling = PSW_max)    
        # Save trained model
        path = f'Ventil_trained_NNs\\tcn_{params["experiment_number"]}.pth'
        torch.save(model.state_dict(), path)
        print(f"Run finished, file saved as: \n {path}")

        # Log parameters
        logging.info(f"hyperparams: {params}")
        logging.info(f"Final train error over whole traj (average over some inits) {average_traj_err_train}")
        logging.info(f"Final test error over whole traj (average over some inits) {err_test_final}")
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
