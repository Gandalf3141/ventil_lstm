import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataloader_fs import *
import os 
from tqdm import tqdm
import logging
from meas_test_func_fs import *
from meas_load_data import *

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss.append(loss.item())
    lr_scheduler.step()
    #print(lr_scheduler.get_last_lr())

    return np.mean(total_loss)



def main(parameters, i):

    params_lstm = parameters

    # Configure logging
    log_file = f'meas_training_lstm{i}.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the LSTM model
    model_lstm = OR_LSTM(input_size=4, hidden_size=params_lstm["h_size"], out_size=2, layers=params_lstm["l_num"], window_size=params_lstm["window_size"]).to(device)

    # Generate input data (the data is normalized and some timesteps are cut off)

    if os.name == "nt":
        path_train_data=r"C:\Users\strasserp\Documents\ventil_lstm\Experiment_Meassurements\Messungen\messdaten_900traj_500steps.csv"
        path_train_data_festo=r"C:\Users\strasserp\Documents\ventil_lstm\Experiment_Meassurements\Messungen\FESTO_traindata.csv"
    else:
        path_train_data=r"/home/rdpusr/Documents/ventil_lstm/Experiment_Meassurements/Messungen/messdaten_900traj_500steps.csv"
        path_train_data_festo=r"/home/rdpusr/Documents/ventil_lstm/Experiment_Meassurements/Messungen/FESTO_traindata.csv"


    #train_data = get_data(path_train_data,num_inits=params_lstm["part_of_data"])
    train_data_festo = get_data(path_train_data_festo,num_inits=params_lstm["part_of_data"])

    #                   !!! 2/3 of the regular training data is dropped!!!
    #train_data_combined = torch.concatenate([train_data, train_data_festo], dim=0)
    train_data_combined = train_data_festo
    print("combined traindata:", train_data_combined.size())
    train_loader_lstm, test_data = get_dataloader(train_data_combined, params_lstm)
    #train_loader_lstm, test_data = get_dataloader(get_data(path_train_data,num_inits=params_lstm["part_of_data"]), params_lstm)

    average_traj_err_train_lstm = []

    #optimizer
    optimizer = torch.optim.AdamW(model_lstm.parameters(), lr = params_lstm["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = params_lstm["T_max"], eta_min=0, last_epoch=-1, verbose='deprecated')

    #Training loop
    for e in tqdm(range(params_lstm["epochs"])):
        
        train_error = train_lstm(train_loader_lstm, model_lstm, optimizer=optimizer, lr_scheduler=lr_scheduler)
        if (e+1) % 50 == 0:
            print("Training error : ", train_error)
        
    # Save trained model
    path_lstm = f'Experiment_Meassurements/not_pretrained_or_lstm_{i}_{params_lstm["experiment_number"]}.pth'

    torch.save(model_lstm.state_dict(), path_lstm)

    print(f"Run finished!")
    print(path_lstm)

    # Log parameters

    logging.info(f"hyperparams lstm: {params_lstm}")
    #logging.info(f"LSTM - Experiment number {params_lstm['experiment_number']}_{average_traj_err_train_lstm}")   
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

if __name__ == '__main__':


    params_lstm2 =    {
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 4000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000)
                        }
    
    
    params_lstm3 =    {
                        "window_size" : 80,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 40,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 3000,
                        "test_every_epochs" : 2,
                        "T_max" : 4000,
                        "experiment_number" : np.random.randint(0,1000)
                        
                        }

    
    param_list = [params_lstm3]

    for i, parameters in enumerate(param_list):

        parameters["percentage_of_data"]  = 0.9
        parameters["cut_off_timesteps"]  = 0
        parameters["part_of_data"]  = 0
        parameters["epochs"]  = 4000
        parameters["test_every_epochs"]  = 100
        parameters["experiment_number"]  = np.random.randint(0,1000)

        main(parameters, i)
