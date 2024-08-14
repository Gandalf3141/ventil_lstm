import torch
from get_data_fs import *
import os
from dataloader_fs import *


def load_data(num_inits):

    if os.name == "nt":
        input_data1 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_01_IV_sprung.csv", num_inits=num_inits)
        input_data2 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_01_IV2.csv", num_inits=num_inits)
        input_data3 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_01_randomwalk.csv", num_inits=num_inits)
        input_data4 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_01_same_u_und_mixed150.csv", num_inits=num_inits)
        input_data5 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_01_randomwalk_stationary_mix.csv", num_inits=num_inits)
        input_data6 = get_data(path = r"C:\Users\strasserp\Documents\ventil_lstm\data_fs\training_data_full_system_train_mixed_constant.csv", num_inits=num_inits)
    else:
        input_data1 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_01_IV_sprung.csv", num_inits=num_inits)
        input_data2 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_01_IV2.csv", num_inits=num_inits)
        input_data3 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_01_randomwalk_stationary_mix.csv", num_inits=num_inits)
        input_data4 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_01_randomwalk.csv", num_inits=num_inits)
        input_data5 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_01_same_u_und_mixed150.csv", num_inits=num_inits)
        input_data6 = get_data(path = r"/home/rdpusr/Documents/ventil_lstm/data_fs/training_data_full_system_train_mixed_constant.csv", num_inits=num_inits)

   
    input_data = torch.cat((input_data1, input_data2, input_data3, input_data4, input_data5, input_data6))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    #input_data = input_data.to(device)

    return input_data

def get_dataloader(input_data, params): 

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
    train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
    np.random.shuffle(train_inits)
    np.random.shuffle(test_inits)
    #train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
    #cut off timesteps at the start
    train_data = input_data[train_inits,params["cut_off_timesteps"]:,:]
    test_data = input_data[test_inits,:,:]

    # dataloader for batching during training
    train_set = custom_simple_dataset(train_data, window_size=params["window_size"])
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], pin_memory=True)

    return train_loader, test_data
