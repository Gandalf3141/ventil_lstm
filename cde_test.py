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

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

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
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
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
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y

######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def unused_get_data_(num_timepoints=100):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    return X, y

def main():

    params =                    {
                                "window_size" : 50,
                                "h_size" : 8,
                                "l_num" : 3,
                                "epochs" : 3,
                                "learning_rate" : 0.001,
                                "part_of_data" : 2, 
                                "percentage_of_data" : 0.8,
                                "batch_size" : 100,
                                "cut_off_timesteps" : 0,
                                "drop_half_timesteps": True
                                }
    
    input_data1, PSW_max = get_data_cde(path = "data\save_data_test_revised.csv", 
                                timesteps_from_data=0, 
                                skip_steps_start = 0,
                                skip_steps_end = 0, 
                                drop_half_timesteps = params["drop_half_timesteps"],
                                normalise_s_w="minmax",
                                rescale_p=False,
                                num_inits=params["part_of_data"])
    input_data = input_data1
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
        train_loader = DataLoader(train_set, batch_size=params["batch_size"], pin_memory=True)  



    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=4, hidden_channels=8, output_channels=2).to(device)
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
    for epoch in range(params["epochs"]):
        
        for x, y in train_loader:

            train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)

            batch_coeffs, batch_y = train_coeffs, y

            pred_y = model(batch_coeffs).squeeze(-1)

            loss = loss_fn(pred_y, batch_y[:, 2:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))    

        if epoch % 2 == 0:            
            test_loss, test_loss_deriv, err_test = test(train_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                    display_plots=False, num_of_inits = 1, set_rand_seed=True, physics_rescaling = PSW_max)
            print('Epoch: {}   Test loss: {}'.format(epoch, err_test.item()))

    print("Training finised!")
    test_loss, test_loss_deriv, err_test = test(test_data.to(device), model, model_type = "neural_cde", window_size=params["window_size"], 
                                                    display_plots=True, num_of_inits = 10, set_rand_seed=True, physics_rescaling = PSW_max)


    path = f'Ventil_trained_NNs\cde{params["experiment_number"]}.pth'
    torch.save(model.state_dict(), path)
    print(f"Run finished, file saved as: \n {path}")


if __name__ == '__main__':
    main()