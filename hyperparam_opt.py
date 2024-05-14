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
from ray.train import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
import ray
import logging
import os
import tempfile

ray.init()

# Use the GPU if available
#torch.set_default_dtype(torch.float64)
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="cpu"
torch.set_default_dtype(torch.float64)

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
        self.act = nn.ReLU()
        # Define LSTM layer
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
        #lstm_out = self.act(lstm_out)
        pred = self.linear(lstm_out)

        return pred, hidden

def train_epoch(input_data, model, weight_decay, future_decay, learning_rate=0.001, ws=0, future=1, timesteps=0, batch_size=0):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(input_data):  

        if ((k+1)%(timesteps/batch_size))*(batch_size + ws + future) > timesteps:
            continue
        # if k%future != 0:
        #     continue
        
        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output, _ = model(inp)
        out = inp[:, :, 1:] + output
        
        inputs = [0]
        outputs = [out]
        loss_future = []

        for t in range(future-1): 
           
            new_combined_inp = torch.cat((label[:, t, 0:1], outputs[t][:,-1,:]), dim=1)
            new_combined_inp = new_combined_inp.view(inp.size(dim=0),1,3)
            if t>0:
                inputs.append(torch.cat((inputs[t-1][: , 1:ws,:], new_combined_inp), dim =1))
            else:
                inputs[0] = torch.cat((inp[: , 1:ws,:], new_combined_inp), dim =1)

            output2, _ = model(inputs[t])
            outputs.append(inputs[t][:, :, 1:] + output2)

            loss_future.append(loss_fn(outputs[t][:,-1,:], label[:, t+1, 1:]))


        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error
        if future<2:
            loss = loss_fn(out[:,-1,:], label[:, 1:])
        else:   
            loss = loss_fn(out[:,-1,:], label[:, 0, 1:])

        #backpropagation
        if future>1:
            for loss_f in loss_future:
            
             loss += future_decay * loss_f

            loss.backward(retain_graph=True)
            optimizer.step()


        total_loss.append(loss.detach().cpu().numpy())

   # return the average error of the next step prediction
    return np.mean(total_loss)

def test(test_data, model, steps=600, ws=10, plot_opt=False, n = 5):

    #test_data = test_dataloader.get_all_data() 
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0
    total_loss = 0

    ids = np.random.randint(0, test_data.size(dim=0), n)
    ids = np.unique(ids)

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


def objective(config):  # ①

    #print("calling objective function with config:", config)
    torch.set_default_dtype(torch.float64)

    #other parameters:
    fixed_params = {
                    "part_of_data" : 0,
                    "percentage_of_data" : 0.8,
                    "future" : 4,
                    "weight_decay" : 1e-5,  
                    "future_decay" : 1,
                    "ls" : 1
                    }

    # Initialize the LSTM model
    model = LSTMmodel(input_size=3, hidden_size=config["hs"], out_size=2, layers=fixed_params["ls"]).to(device)
    # Generate input data (the data is normalized and some timesteps are cut off)
    input_data = get_data(path = r"C:\Users\StrasserP\Documents\Python Projects\ventil_lstm\save_data_test3.csv", 
                            timesteps_from_data=0, 
                            skip_steps_start = 0,
                            skip_steps_end = 0, 
                            drop_half_timesteps = False,
                            normalise_s_w=True,
                            rescale_p=False,
                            num_inits=fixed_params["part_of_data"])
    cut_off_timesteps = 100

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*fixed_params["percentage_of_data"])
    train_inits = np.random.randint(0,len(input_data), num_of_inits_train)
    train_inits = np.unique(train_inits)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
    # make sure we really get the specified percentage of training data..
    if fixed_params["percentage_of_data"] < 0.99: 
            while len(train_inits) < num_of_inits_train:
                i = np.random.randint(0,len(test_inits),1)[0]
                train_inits = np.append(train_inits,test_inits[i])
                test_inits = np.delete(test_inits, i)

    train_data = input_data[train_inits,:input_data.size(dim=1)-cut_off_timesteps,:]
    test_data = input_data[test_inits,:,:]

    data_set  = CustomDataset(train_data, window_size=config["ws"], future=config["fu"])
    train_dataloader = DataLoader(data_set, batch_size=config["bs"],drop_last=True)#, pin_memory=True)

    epochs=40


        # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
    
    for e in range(epochs):
        
        train_epoch(train_dataloader, model, fixed_params["weight_decay"], fixed_params["future_decay"], learning_rate=config["lr"], ws=config["ws"], future=config["fu"],
                     timesteps=train_data.size(dim=1), batch_size=config["bs"])  # Train the model
        _,_, acc = test(test_data, model, steps=test_data.size(dim=1), ws=config["ws"], plot_opt=False, n = 100)  # Compute test accuracy

        if (e+1)%5 == 0:
            
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
                "ws": tune.randint(lower=1, upper=17),
                "bs": tune.randint(lower=8,upper=128),
                "hs": tune.randint(lower=4, upper=33),
                "fu" : tune.choice([2,4,8,16,32]),
               # "ls": tune.randint(lower=1, upper=4)
               }

algo = OptunaSearch()  # ②
algo = ConcurrencyLimiter(algo, max_concurrent=32)

scheduler =  ASHAScheduler(max_t = 4, grace_period = 1, reduction_factor=2)

tuner = tune.Tuner(  # ③
    objective,
    tune_config=tune.TuneConfig(
        num_samples=200,
        metric="mean_accuracy",
        mode="min",
        search_alg=algo,
        scheduler=scheduler
    ),
    #run_config=train.RunConfig(
    # stop={"training_iteration": 5},
    #),
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