# Importing necessary libraries
from matplotlib import legend
import pandas as pd
import torch
from torch import nn
import os
import numpy as np
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

#Define the LSTM model class

# Use the GPU if available
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

class custom_simple_dataset(Dataset):
 
 
    def __init__(self, data, window_size):
 
        self.data = data
        self.ws = window_size
 
    def __len__(self):
        return self.data.size(0)
 
    def __getitem__(self, idx):
 
        inp = self.data[idx, :, :]
        label = self.data[idx, self.ws:, 1:].clone()

        return inp, label


def fr(v):

#parBoost_gen.d_b  = 3.2;                    % Dämpfungskonste Booster in [Ns/m]  Update AP: 2021Feb alter Wert 33.184
#parBoost_gen.F_c  = 0.5;                      % Coulombreibkraft Booster in [N] Update AP: 2021Feb alter Wert: 1.53;

    d = 3.2
    F_c  = 0.5

    #fr = - d * v - F_c * torch.sign(v)
    fr = - d * v - F_c * torch.tanh(v/0.1)

    return fr

def fk(s,v):

# % Paramter Kontaktmodell
# % Unterer Anschlag
# parBoost_gen.c_bwl = 166e3;          % Federkonstante unterer Anschlag in [N/m]
# parBoost_gen.d_bwl = 474;            % Dämpfungskonstante unterer Anschlag in [Ns/m]
# parBoost_gen.s_0bwl = 0.3e-4;        % Kontaktpunkt unterer Anschlag in [m] 
# % Oberer Anschlag
# parBoost_gen.c_bwu = 166e3;                         % Federkonstante oberer Anschlag in [N/m]
# parBoost_gen.d_bwu = 474;                           % Dämpfungskonstante oberer Anschlag in [Ns/m]
# parBoost_gen.s_0bwu = parBoost_gen.s_b_max - 0.12e-4;     % Kontaktpunkt oberer Anschlag in [m]
#parBoost_gen.s_b_max = 0.6e-3;                                      % Maximaler Hub in [m]

    s_u =  0.6e-3 - 0.12e-4
    c_u = 166e3
    d_u = 474

    c_l = 166e3
    s_l =  0.3e-4
    d_l = 474


    if s_u <= s:
        
        fk = -c_u * (s - s_u) - d_u * v * (s - s_u)

    elif s <= s_l:

        fk = c_l * (s - s_l) - d_l * v * (s - s_l)

    else:
        fk = 0

    return fk

def ODE_right_side(x, pressure, physics_rescaling=None, no_fk=False, no_fr=False):

    #rescale to physical units

   # x[:, :,0] = x[:, :,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
   # x[:, :,1] = x[:, :,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
   # pressure[:, :,0] = pressure[:, :,0]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

    x_dt = torch.zeros_like(x[:,:,0:1]) # write s' = v (v from real data)  
    #          v'  = 1/m ( A * ( p - p0) - c * (s - s0) + fr(v) + fk(s,v) )  

    # andere Formulierung: fs = -(c_b*s_b - F_cb0) statt -c_b *(s_b - s_0b);

    #          v'  = 1/m ( A * ( p - p0) - (c_b*s_b - F_cb0) + fr(v) + fk(s,v) )  


    m = 1.8931e-3                         # 1.8931e-3;              % Masse Booster in [kg]
    A = 0.5*(71.0526e-6 + 78.9793e-6 )  # 0.5*(parBoost_gen.A_b_closed + parBoost_gen.A_b_open);   % Mittlere Fläche Booster in [m²]                                
                                            # % Anfangswerte der Boostereinheit
    p0 = 1e5                                # parBoost_gen.s_b_0 = 3e-4;
    s0 = 3e-4                               # parBoost_gen.p_b_0 = 1e5;
    c = 16.5e3                              # parBoost_gen.c_b  = 16.5e3;                 % Federkonstante Booster in [N/m]
    F_cb0 = -4.3                            #parBoost_gen.F_cb0 = -4.3;                    % Federvorspannkraft (Kraft der Feder bei sb=0) [N]

    for i in range(x_dt.size(dim=0)):
        for time in range(x_dt.size(dim=1)):

            p = pressure[i, time, 0] * (physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]
            s =  x[i, time, 0] * (physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
            v = x[i, time, 1] * (physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
            
            if no_fr:
                x_dt[i, time, 0:1] =  1/m * ( A * ( p - p0) - (c * s - F_cb0) + fk(s,v) ) 
            
            elif no_fk and not no_fr:
                x_dt[i, time, 0:1] =  1/m * ( A * ( p - p0) - (c * s - F_cb0) + fr(v)) 
            
            elif no_fk and no_fr:
                x_dt[i, time, 0:1] =  1/m * ( A * ( p - p0) - (c * s - F_cb0)) 

            else:
                x_dt[i, time, 0:1] =  1/m * ( A * ( p - p0) - (c * s - F_cb0) + fr(v) + fk(s,v) ) 

    return x_dt

def train(input_data, model, weight_decay, learning_rate=0.001, ws=0, PSW_max=0, physics_loss_weight = 0.0001, phy_options=[False, False]):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
 
    model.train()
    total_loss = []
    total_physics_loss = []

    for k, (x,y) in enumerate(input_data):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
        
        output, _, derivative_sv = model(x)

        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(output, y)

        # calc physics loss
        physics_loss = None
        if physics_loss_weight != 0:
        
         pressure = x[:, ws:, 0:1] # Anfangswerte sind abgeschnitten weil len(output) == len(x) - windowsize 

         a_net = ODE_right_side(output, pressure, PSW_max, phy_options[0], phy_options[1])
         a_true = ODE_right_side(y, pressure, PSW_max, phy_options[0], phy_options[1])
       
         physics_loss = physics_loss_weight * loss_fn(a_net, a_true)

        losses = loss.copy()

        if physics_loss != None:
         losses += physics_loss

        losses.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
        if physics_loss != None:
         total_physics_loss.append(physics_loss.detach().cpu().numpy())

   # return the average error of the next step prediction

   
    return np.mean(total_loss), np.mean(total_physics_loss)

def main():
                
    parameter_configs  = [                           
                                {
                                "experiment_number" : 2,
                                "window_size" : 16,
                                "h_size" : 8,
                                "l_num" : 3,
                                "epochs" : 1000,
                                "learning_rate" : 0.0008,
                                "part_of_data" : 0, 
                                "weight_decay" : 0,
                                "percentage_of_data" : 0.8,
                                "future_decay"  : 0.5,
                                "batch_size" : 20,
                                "future" : 10,
                                "cut_off_timesteps" : 100,
                                "drop_half_timesteps": True,
                                "physics_loss_weight" : 1e-5,
                                "phy_options" : [True, True]
                                }        
    ]
                  

    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k

    for k, params in enumerate(parameter_configs):

        # Configure logging
        log_file = 'training_pinn.log'
        filemode = 'a' if os.path.exists(log_file) else 'w'
        logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

            # Initialize the LSTM model
        model = LSTMmodel(input_size=3, hidden_size=params["h_size"], out_size=2, layers=params["l_num"], window_size=params["window_size"]).to(device)

        path = "working_networks\OR_lstm_16_8_3_best_V2.pth"
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))

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
        input_data = input_data1
        print(input_data.size())

        #Split data into train and test sets
        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
        np.random.shuffle(train_inits)
        np.random.shuffle(test_inits)
        train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
        test_data = input_data[test_inits,:,:]

        # dataloader for batching during training
        train_set = custom_simple_dataset(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=params["batch_size"], pin_memory=True)

        losses = []
        average_traj_err_train = []
        average_traj_err_test = []

        
        #Error before retraining the network
        test_loss, test_loss_deriv, error_baseline = test(test_data.to(device), model, model_type = "or_lstm", window_size=params["window_size"], 
                                                          display_plots=False, num_of_inits = 200, set_rand_seed=True, physics_rescaling = PSW_max)

        for e in tqdm(range(params["epochs"])):
            

            loss_epoch, physics_loss_epochs = train(train_loader, model, params["weight_decay"], learning_rate= params["learning_rate"],
                                                        ws=params["window_size"], PSW_max=PSW_max,
                                                        physics_loss_weight=params["physics_loss_weight"], phy_options = params["phy_options"])
            losses.append(loss_epoch)

            # Every few epochs get the error MSE of the true data
            # compared to the network prediction starting from some initial conditions
            if (e+1)%100 == 0:
                
                print(f"Epoch {e} , Train loss:", loss_epoch)
                print(f"Epoch {e} , physics loss:", physics_loss_epochs)
                
                #_,_, err_train = test(train_data, model, steps=train_data.size(dim=1), ws=params["window_size"], plot_opt=False, test_inits=len(train_inits), n = 20, PSW_max=PSW_max)
                test_loss, test_loss_deriv, err_train = test(train_data.to(device), model, model_type = "or_lstm", window_size=params["window_size"],
                                                              display_plots=False, num_of_inits = 50, set_rand_seed=True, physics_rescaling = PSW_max)


                average_traj_err_train.append(err_train)
                average_traj_err_test.append(err_train)

                #print(f"Average error over full trajectories: training data : {err_train}")
                print(f"Average error over full trajectories: training data : {err_train}")
        
        test_loss, test_loss_deriv, err_test_final = test(test_data.to(device), model, model_type = "or_lstm", window_size=params["window_size"], 
                                                          display_plots=False, num_of_inits = 200, set_rand_seed=True, physics_rescaling = PSW_max)

        # Save trained model
        path = f'Ventil_trained_NNs\OR_lstm_PINN_retrained{params["experiment_number"]}.pth'
        torch.save(model.state_dict(), path)
        print("Training with pretrained model finished!")
        print("Baseline erorr: ", error_baseline)
        print("New Error after retraining with PhysLoss: ", err_test_final)
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
# comment