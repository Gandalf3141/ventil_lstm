import torch
from get_data_fs import *
from NN_classes_fs import *
from dataloader_fs import *
import os 
from tqdm import tqdm
import logging
from test_func_fs import *

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("this device is available : ", device)

# train function
def train_mlp(loader, model, learning_rate=0.001):
 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    model.train()
    total_loss = []
  
    for x,y in loader:  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
 
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

params_mlp =    {
                        "window_size" : 20,
                        "h_size" : 24,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "act_fn" : "relu",
                        "nonlin_at_out" : None, #None if no nonlinearity at the end

                        "percentage_of_data" : 0.9,
                        "cut_off_timesteps" : 300,
                        "part_of_data" : 10,
                        "epochs" : 10,
                        "test_every_epochs" : 3,
                        "input_channels" : 5,
                        "output" : 3,
                        "experiment_number" : np.random.randint(0,1000)
                    }

# Configure logging
log_file = 'training_fullsystem.log'
filemode = 'a' if os.path.exists(log_file) else 'w'
logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the MLP model

model_mlp = OR_MLP(input_size=5*params_mlp["window_size"], hidden_size=params_mlp["h_size"], l_num=params_mlp["l_num"], output_size=3,
                    act_fn=params_mlp["act_fn"], act_at_end = None, timesteps=params_mlp["window_size"]).to(device)
# Generate input data (the data is normalized and some timesteps are cut off)
input_data1 = get_data(path = "data_fs/training_data_full_system_01_IV_sprung.csv", num_inits=params_mlp["part_of_data"])
input_data2 = get_data(path = "data_fs/training_data_full_system_01_randomwalk.csv", num_inits=params_mlp["part_of_data"])
input_data3 = get_data(path = "data_fs/training_data_full_system_01_IV2.csv", num_inits=params_mlp["part_of_data"])

input_data = torch.cat((input_data1, input_data2, input_data3))
print(input_data.size())

#Split data into train and test sets
np.random.seed(1234)
num_of_inits_train = int(len(input_data)*params_mlp["percentage_of_data"])
train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
np.random.shuffle(train_inits)
np.random.shuffle(test_inits)
train_data = input_data[train_inits,:input_data.size(dim=1)-params_mlp["cut_off_timesteps"],:]
test_data = input_data[test_inits,:,:]

# dataloader for batching during training
train_set_mlp = custom_simple_dataset(train_data, window_size=params_mlp["window_size"])
train_loader_mlp = DataLoader(train_set_mlp, batch_size=params_mlp["batch_size"], pin_memory=True)

average_traj_err_train_mlp = []

#Training loop
for e in tqdm(range(params_mlp["epochs"])):
    
    train_error = train_mlp(train_loader_mlp, model_mlp, learning_rate=params_mlp["learning_rate"])
    if (e+1) % 50 == 0:
        print("Training error : ", train_error)

    # Every few epochs get the error MSE of the true data
    # compared to the network prediction starting from some initial conditions
    if (e+1)%params_mlp["test_every_epochs"] == 0:
        err_test_mlp = test(test_data.to(device), model_mlp, model_type="mlp", window_size=params_mlp["window_size"], display_plots=False, numb_of_inits = 10)
        average_traj_err_train_mlp.append(err_test_mlp)
        print(f"Average error over full trajectories: test data MLP: {err_test_mlp}")
        
# Save trained model
path_mlp = f'Full_System/or_mlp{params_mlp["experiment_number"]}.pth'

torch.save(model_mlp.state_dict(), path_mlp)

print(f"Run finished!")

# Log parameters

logging.info(f"hyperparams mlp: {params_mlp}")
logging.info(f"MLP - Experiment number {params_mlp['experiment_number']}_{average_traj_err_train_mlp}")   
logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
logging.info("\n")
