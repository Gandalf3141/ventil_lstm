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
#from NN_classes import *

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)


######################
# A CDE model looks like

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_width=128):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_width  = hidden_width

        self.linear1 = torch.nn.Linear(hidden_channels, self.hidden_width)
        self.linear2 = torch.nn.Linear(self.hidden_width, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        # try without?        
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_width, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, hidden_width)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func
                              ,t=X.interval, adjoint=False, backend='torchdiffeq',atol = 1e-4, rtol = 1e-4)#, options=dict(jump_t=X.grid_points),)
                                #method='rk4', 
                                #options=dict(step_size=2e-4),
                                
                                #adjoint=False, atol = 1e-4, rtol = 1e-4)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y




def main():

    parameter_configs =        [  

                            {
                            "window_size" : 50,
                            "h_size" : 16,
                            "h_width" : 128,
                            "batch_size" : 300,
                            "learning_rate" : 0.0005
                            }
                            # this worked ... kind of
                            #       "window_size" : 50,
                            # "h_size" : 8,
                            # "h_width" : 128,
                            # "batch_size" : 500,
                            # "learning_rate" : 0.0005
                        ]
    
    
    for k, d in enumerate(parameter_configs):
        d["experiment_number"] = k
        d["epochs"] = 200
        d["part_of_data"] = 0
        d["percentage_of_data"] = 0.7
        d["cut_off_timesteps"] = 0
        d["future"] = 1
        d["drop_half_timesteps"] = True 

    for params in parameter_configs:

        # Generate input data (the data is normalized and some timesteps are cut off)
        input_data1, PSW_max = get_data_cde(path = "data\save_data_test_revised.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        input_data2, PSW_max = get_data_cde(path = "data\save_data_test5.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        input_data3, PSW_max = get_data_cde(path = "data\Testruns_from_trajectory_generator_t2_t6_revised.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])

        input_data = torch.cat((input_data1, input_data2, input_data3)).to(device)
    
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
        
        model = NeuralCDE(input_channels=4, hidden_channels=params["h_size"], hidden_width = params["h_width"], 
                          output_channels=2, interpolation="cubic").to(device)
        
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
                #train_coeffs = torchcde.linear_interpolation_coeffs(x)
                batch_coeffs, batch_y = train_coeffs.to(device), y.to(device)
            
                pred_y = model(batch_coeffs).squeeze(-1)
                #out = x[:,-1,2:].squeeze(-1)+ pred_y
                out = x[:,-1,2:] + pred_y
                loss = loss_fn(out, batch_y[:, 2:])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch: {}   Training loss (next step): {}'.format(epoch, loss.item()))    

            if (epoch+1) % 50 == 0:            
                test_loss, test_loss_deriv, err_test = test(test_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                        display_plots=False, num_of_inits = 1, set_rand_seed=True, physics_rescaling = PSW_max)
                print('Epoch: {}   Test loss (MSE over whole Traj.): {}'.format(epoch, err_test.item()))

        path = f'Ventil_trained_NNs\cde{params["experiment_number"]}.pth'
        torch.save(model.state_dict(), path)
        print(f"Run finished, file saved as: \n {path}")

        test_loss, test_loss_deriv, err_test = test(test_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                        display_plots=False, num_of_inits = 2, set_rand_seed=True, physics_rescaling = PSW_max)
        print("Training finised! Final error:", err_test)        


if __name__ == '__main__':
    main()