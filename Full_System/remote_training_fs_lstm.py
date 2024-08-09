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

def train_lstm(input_data, model,  optimizer, lr_scheduler):
 
    loss_fn = nn.MSELoss()
 
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
    lr_scheduler.step()

    return np.mean(total_loss)



def main(parameters, i):

    params_lstm = parameters

    # Configure logging
    log_file = f'training_fullsystem_lstm{i}.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the LSTM model
    model_lstm = OR_LSTM(input_size=5, hidden_size=params_lstm["h_size"], out_size=3, layers=params_lstm["l_num"], window_size=params_lstm["window_size"]).to(device)

    # Generate input data (the data is normalized and some timesteps are cut off)
    input_data1 = get_data(path = "data_fs/training_data_full_system_01_IV_sprung.csv", num_inits=params_lstm["part_of_data"])
    input_data2 = get_data(path = "data_fs/training_data_full_system_01_randomwalk.csv", num_inits=params_lstm["part_of_data"])
    input_data3 = get_data(path = "data_fs/training_data_full_system_01_IV2.csv", num_inits=params_lstm["part_of_data"])
    input_data4 = get_data(path = "data_fs/training_data_full_system_01_same_u_und_mixed150.csv", num_inits=params_lstm["part_of_data"])
    input_data = torch.cat((input_data1, input_data2, input_data3, input_data4))
    print(input_data.size())

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*params_lstm["percentage_of_data"])
    train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
    np.random.shuffle(train_inits)
    np.random.shuffle(test_inits)
    train_data = input_data[train_inits,:input_data.size(dim=1)-params_lstm["cut_off_timesteps"],:]
    test_data = input_data[test_inits,:,:]

    # dataloader for batching during training
    train_set_lstm = custom_simple_dataset(train_data, window_size=params_lstm["window_size"])
    train_loader_lstm = DataLoader(train_set_lstm, batch_size=params_lstm["batch_size"], pin_memory=True)

    average_traj_err_train_lstm = []

    #optimizer
    optimizer = torch.optim.AdamW(model_lstm.parameters(), lr = params_lstm["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = params_lstm["T_max"], eta_min=0, last_epoch=-1, verbose='deprecated')

    #Training loop
    for e in tqdm(range(params_lstm["epochs"])):
        
        train_error = train_lstm(train_loader_lstm, model_lstm, optimizer=optimizer, lr_scheduler=lr_scheduler)
        if (e+1) % 50 == 0:
            print("Training error : ", train_error)

        # Every few epochs get the error MSE of the true data
        # compared to the network prediction starting from some initial conditions
        if (e+1)%params_lstm["test_every_epochs"] == 0:
            err_test_lstm = test(test_data.to(device), model_lstm, model_type="lstm", window_size=params_lstm["window_size"], display_plots=False, numb_of_inits = 10)
            average_traj_err_train_lstm.append(err_test_lstm)
            print(f"Average error over full trajectories: test data LSTM: {err_test_lstm}")
            
    # Save trained model
    path_lstm = f'Full_System/or_lstm_{i}_{params_lstm["experiment_number"]}.pth'

    torch.save(model_lstm.state_dict(), path_lstm)

    print(f"Run finished!")

    # Log parameters

    logging.info(f"hyperparams lstm: {params_lstm}")
    logging.info(f"LSTM - Experiment number {params_lstm['experiment_number']}_{average_traj_err_train_lstm}")   
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

if __name__ == '__main__':
    
    params_lstm_basic =    {
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "T_max" : 100,
                    }

    params_lstm =    {  
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "T_max" : 20,

                        "experiment_number" : np.random.randint(0,1000)}
   
   
    param_list = [params_lstm, params_lstm_basic]

    for i, parameters in enumerate(param_list):

        parameters["percentage_of_data"]  = 0.8
        parameters["cut_off_timesteps"]  = 0
        parameters["part_of_data"]  = 100
        parameters["epochs"]  = 100
        parameters["test_every_epochs"]  = 50
        parameters["experiment_number"]  = np.random.randint(0,1000)

        main(parameters, i)
