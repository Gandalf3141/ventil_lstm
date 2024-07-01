######################
# So you want to train a Neural CDE model?
# Let's get started!
######################

import math
import torch
import torchcde
from get_data import *
from dataloader import *
from test_function import *
from tqdm import tqdm
from NN_classes import *

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)



def main():

    parameter_configs =        [  

                                {
                            "experiment_number" : 0,
                            "window_size" : 10,
                            "h_size" : 8,
                            "h_width" : 8,
                            "epochs" : 100,
                            "learning_rate" : 0.001,
                            "part_of_data" : 100, 
                            "percentage_of_data" : 0.7,
                            "batch_size" : 50,
                            "cut_off_timesteps" : 100,
                            "drop_half_timesteps": True
                            }
                    ]
    
    
    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k

    for params in parameter_configs:

        input_data1, PSW_max = get_data_cde(path = "data\save_data_test_revised.csv", 
                                    timesteps_from_data=0, 
                                    skip_steps_start = 0,
                                    skip_steps_end = 0, 
                                    drop_half_timesteps = params["drop_half_timesteps"],
                                    normalise_s_w="minmax",
                                    rescale_p=False,
                                    num_inits=params["part_of_data"])
        input_data = input_data1.to(device)

        #cols = time_cols, pb_cols, sb_cols, wb_cols

            #Split data into train and test sets
        np.random.seed(1234)
        num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
        train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
        test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
        np.random.shuffle(train_inits)
        np.random.shuffle(test_inits)
        train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
        test_data = input_data[test_inits,:,:]


        train_set = CustomDataset_cde(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=params["batch_size"])  
        if device == "cuda:0":
            print("gpu dataloader")
            train_loader = DataLoader(train_set, batch_size=params["batch_size"])  

        print(train_data.size(), test_data.size())
        ######################
        # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
        # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
        # output_channels=1 because we're doing binary classification.
        ######################
        model = NeuralCDE(input_channels=4, hidden_channels=params["h_size"], hidden_width = params["h_width"], output_channels=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = torch.nn.MSELoss()

        ######################
        # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
        # The resulting `train_coeffs` is a tensor describing the path.
        # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
        ######################
            # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

            # train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
            # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        for epoch in tqdm(range(params["epochs"])):
            
            for x, y in train_loader:
                x = x.to(device)
                train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
                batch_coeffs, batch_y = train_coeffs.to(device), y.to(device)
            
                pred_y = model(batch_coeffs).squeeze(-1)
                #out = x[:,-1,2:].squeeze(-1)+ pred_y
                out = x[:,-1,2:] + pred_y
                loss = loss_fn(out, batch_y[:, 2:])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            #print('Epoch: {}   Training loss (next step): {}'.format(epoch, loss.item()))    

            if (epoch+1) % 50 == 0:            
                test_loss, test_loss_deriv, err_test = test(test_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                        display_plots=False, num_of_inits = 1, set_rand_seed=True, physics_rescaling = PSW_max)
                print('Epoch: {}   Test loss (MSE over whole Traj.): {}'.format(epoch, err_test.item()))

        path = f'Ventil_trained_NNs\cde{params["experiment_number"]}.pth'
        torch.save(model.state_dict(), path)
        print(f"Run finished, file saved as: \n {path}")

        test_loss, test_loss_deriv, err_test = test(test_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                        display_plots=False, num_of_inits = 20, set_rand_seed=True, physics_rescaling = PSW_max)
        print("Training finised! Final error:", err_test)        


if __name__ == '__main__':
    main()