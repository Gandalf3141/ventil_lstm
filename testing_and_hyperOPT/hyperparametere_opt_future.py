# Importing necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from icecream import ic
from tqdm import tqdm
from get_data import *
from dataloader import *
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.train import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
import ray
import logging
import os
import tempfile
from NN_classes import *

ray.init()

# Use the GPU if available
#torch.set_default_dtype(torch.float64)
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="cpu"
torch.set_default_dtype(torch.float64)

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

def train_epoch(input_data, model, weight_decay, learning_rate=0.001, ws=0):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
 
    model.train()
    total_loss = []
  
    for k, (x,y) in enumerate(input_data):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
        
        output, _ = model(x)
  
        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
 
   # return the average error of the next step prediction
    return np.mean(total_loss)

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

        
        x = x.view(1,x.size(dim=0), x.size(dim=1))

        if i not in ids:
            continue
 
        with torch.inference_mode():
 
            pred = torch.zeros((steps, 3), device=device)
 
            if ws > 1:
                pred[0:ws, :] = x[0, 0:ws, :]
                pred[:, 0] = x[0, :, 0]
 
            else:
                pred[0, :] = x[0, 0, :]
                pred[:, 0] = x[0, :, 0]
 

            out, _ = model(x)
            pred[ws:,1:] = out
           
            test_loss += loss_fn(pred[ws:, 1], x[0, ws:, 1]).detach().cpu().numpy()
            test_loss_deriv += loss_fn(pred[ws:, 2], x[0, ws:, 2]).detach().cpu().numpy()
            total_loss += loss_fn(pred[ws:, 1:], x[0, ws:, 1:]).detach().cpu().numpy()

            #scale back:
            if PSW_max != 0:
                pred[:,0] = pred[:,0]*PSW_max[0]
                pred[:,1] = pred[:,1]*PSW_max[1]
                pred[:,2] = pred[:,2]*PSW_max[2]
                x[0, :,0] = x[0, :,0]*PSW_max[0]
                x[0, :,1] = x[0, :,1]*PSW_max[1]
                x[0, :,2] = x[0, :,2]*PSW_max[2]

            if plot_opt:
                figure , axs = plt.subplots(1,3,figsize=(16,9))
           
                axs[0].plot(pred.detach().cpu().numpy()[:, 1], color="red", label="pred")
                axs[0].plot(x.detach().cpu().numpy()[0, :, 1], color="blue", label="true", linestyle="dashed")
                axs[0].set_title("position")
                axs[0].grid()
                axs[0].legend()
 
                axs[1].plot(pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
                axs[1].plot(x.detach().cpu().numpy()[0, :, 2], color="blue", label="true", linestyle="dashed")
                axs[1].set_title("speed")
                axs[1].grid()
                axs[1].legend()
 
                axs[2].plot(x.detach().cpu().numpy()[0, :,0], label="pressure")
                axs[2].set_title("pressure")
                axs[2].grid()
                axs[2].legend()
 
                plt.grid(True)
                plt.legend()
                plt.show()
           
    return np.mean(test_loss), np.mean(test_loss_deriv), np.mean(total_loss)


def objective(config):  # ①

    #print("calling objective function with config:", config)
    torch.set_default_dtype(torch.float64)

    #other parameters:
    fixed_params = {
                    "part_of_data" : 100,
                    "percentage_of_data" : 0.8,
                    "weight_decay" : 1e-5,  
                    "future_decay" : 0.5,
                    #"ls" : 1,
                    "fu" : 1,
                    "cut_off_timesteps" : 100,
                    "drop_halt_timesteps" : True
                    }

    # Initialize the LSTM model
    model = LSTMmodel(input_size=3, hidden_size=config["hs"], out_size=2, layers=config["ls"], window_size=config["ws"]).to(device)
    # Generate input data (the data is normalized and some timesteps are cut off)

    input_data, PSW_max = get_data(path = r"C:\Users\StrasserP\Documents\Python Projects\ventil_lstm\save_data_test4.csv", 
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = fixed_params["drop_halt_timesteps"],
                            normalise_s_w="minmax",
                            rescale_p=False,
                            num_inits=fixed_params["part_of_data"])
    
    cut_off_timesteps = fixed_params["cut_off_timesteps"]

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*fixed_params["percentage_of_data"])
    train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])

    train_data = input_data[train_inits,:input_data.size(dim=1)-cut_off_timesteps,:]
    #test_data = input_data[test_inits,:,:]

    data_set  = custom_simple_dataset(train_data, window_size=config["ws"])
    train_dataloader = DataLoader(data_set, batch_size=config["bs"], drop_last=True)#, pin_memory=True)

    epochs=55

        # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
    
    for e in range(epochs):
        
        train_epoch(train_dataloader, model, fixed_params["weight_decay"], learning_rate=config["lr"], ws=config["ws"]) 
         # Train the model
        _,_, acc = test(input_data[test_inits,:,:], model, steps=input_data.size(dim=1), ws=config["ws"], plot_opt=False, n = 100, test_inits=len(train_data), PSW_max=0)  # Compute test accuracy
        
        if acc < 0.01:
            logging.info(f"logged config because error was small ({acc}) config: {config}")
        if (e+1)%25 == 0:
            
            train.report({"mean_accuracy": acc}, checkpoint=None)  # Report to Tune
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"mean_accuracy": acc},
                    checkpoint=checkpoint,
            )

#parameters to optimise:
# config : "lr", "ws", "bs", "hs", "ls"
#         learning rate, window size, batch size, hidden size, number of layers

search_space = {"lr": tune.loguniform(1e-4, 1e-2),
                "ws": tune.choice([2,4,8,16]),
                "bs": tune.choice([10, 20, 64,128,256, 400]),
                "hs": tune.randint(lower=4, upper=12),
                #"fu" : tune.choice([2,4,8,16,32]),
                "ls": tune.choice([1,2,3,4,5])
               }

algo = OptunaSearch(metric="mean_accuracy", mode="min")  # ②
#algo = BayesOptSearch(metric="mean_accuracy", mode="min", utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
#algo = ConcurrencyLimiter(algo, max_concurrent=32)

scheduler = AsyncHyperBandScheduler()

tuner = tune.Tuner(  # ③
    objective,
    tune_config=tune.TuneConfig(
       
        metric="mean_accuracy",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=5
    ),
    run_config=train.RunConfig(
     #stop={"training_iteration": 5},
     name="my_exp"
    ),
    param_space=search_space,   
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)

# Configure logging
log_file = 'training.log'
filemode = 'a' if os.path.exists(log_file) else 'w'
logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log parameters
logging.info(f"config output: {results.get_best_result().config}")


# find results in :
# ~/ray_results