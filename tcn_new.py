import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from get_data import *
from dataloader import *
from test_function import *
from NN_classes import TCN
import logging
import os
from tqdm import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
print(device)

def train(input_data, model, weight_decay=0, learning_rate=0.001):


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(input_data):  # inp = (u, x) label = x

        inp=inp.to(device)
        label=label.to(device)
    
        inp = inp.transpose(1,2)

        # Predict one timestep :
        output = model(inp)

        out = inp[:,1:,-1:].squeeze(-1) + output       
        # reset the gradient    
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out, label[:, 1:])
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss.append(loss.detach().cpu().numpy())

    
# return the average error of the next step prediction
    return np.mean(total_loss)

# set some parameters for learning 
parameter_configs =       [     

                        #         best
                        #     {
                        #     "window_size" : 25,
                        #     "learning_rate" : 0.001,
                        #     "batch_size" : 20,
                        #     "cut_off_timesteps" : 0,

                        #     "n_hidden" : 5,
                        #     "levels" : 4,
                        #     "kernel_size" : 7,
                        #     "dropout" : 0
                        # },
    
                      {
                        "window_size" : 25,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "cut_off_timesteps" : 0,

                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "dropout" : 0
                    },
                    {
                        "window_size" : 10,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "cut_off_timesteps" : 0,

                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "dropout" : 0
                    },
                    {
                        "window_size" : 25,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "cut_off_timesteps" : 0,

                        "n_hidden" : 4,
                        "levels" : 6,
                        "kernel_size" : 8,
                        "dropout" : 0.01
                    }
                                          
                          
                ]

for k, d in enumerate(parameter_configs):
    d["experiment_number"] = k
    d["epochs"] = 200
    d["input_channels"] = 3
    d["output"] = 2
    d["part_of_data"] = 100
    d["percentage_of_data"] = 0.7
    d["future"] = 1
    d["drop_half_timesteps"] = True

for k, params in enumerate(parameter_configs):

    # Configure logging
    log_file = 'training_tcn.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
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


    input_channels = params["input_channels"]
    output = params["output"]
    num_channels = [params["n_hidden"]] * params["levels"]
    kernel_size = params["kernel_size"]
    dropout = params["dropout"]

    model = TCN(input_channels, output, num_channels, kernel_size=kernel_size, dropout=dropout).to(device)



    for i in tqdm(range(params["epochs"])):
        err_train = train(train_dataloader, model)
        if (i+1) % 50 ==0:
            _, _, err_test = test(train_data.to(device), model=model, model_type="tcn", window_size=params["window_size"],
                            display_plots=False, num_of_inits = 10, set_rand_seed=True, physics_rescaling = PSW_max, additional_data=None)
            
            print("train", err_train)
            print("test",err_test)


    path = f'Ventil_trained_NNs\\tcn_{params["experiment_number"]}.pth'
    torch.save(model.state_dict(), path)
    print(f"Run finished, file saved as: \n {path}")

    # Log parameters
    logging.info(f"hyperparams: {params}")
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")