
# Importing necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torchcde
import time



def plot_results(x, pred, pred_next_step=None, physics_rescaling=None, additional_data=None):

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    if pred.dim() == 3:
        pred = pred.view(pred.size(dim=1), pred.size(dim=2))
    if pred_next_step != None:
        if pred_next_step.dim() == 3:
            pred_next_step = pred_next_step.view(pred_next_step.size(dim=1), pred_next_step.size(dim=2))

        #scale back:    
    if physics_rescaling != None:

        # we invert:
        # x = (x - xmin)/(xmax - xmin)
        # x * (xmax - xmin) + xmin

        pred[:,0] = pred[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
       # pred[:,0] = pred[:,0]/1e5
        pred[:,1] = pred[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
        pred[:,2] = pred[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]
        x[:,0] = x[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
       # x[:,0] = x[:,0]/1e5
        x[:,1] = x[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
        x[:,2] = x[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

        if additional_data != None:
            for i in range(additional_data.size(dim=0)):
                additional_data[i,:,0] = additional_data[i,:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
                additional_data[i,:,1] = additional_data[i,:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
                additional_data[i,:,2] = additional_data[i,:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

    #figure , axs = plt.subplots(1,3,figsize=(20,8))
    figure , axs = plt.subplots(3,1, figsize=(16,9))
    figure.tight_layout(pad=5.0)

    greek_letterz=[chr(code) for code in range(945,970)]
    mu = greek_letterz[11]

    stepsize = 2e-5
    time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

    if pred_next_step != None:
        axs[0].plot(time, pred_next_step.detach().cpu().numpy()[:, 1], color="green", label="next step from data")
        axs[1].plot(time, pred_next_step.detach().cpu().numpy()[:, 2], color="green", label="next step from data")

    axs[0].plot(time, pred.detach().cpu().numpy()[:, 1], color="red", label="pred")
    axs[0].plot(time, x.detach().cpu().numpy()[:, 1], color="blue", label="true", linestyle="dashed")

    if additional_data != None:
        for i in range(additional_data.size(dim=0)):
           names = ["simulink", "Hub im Regler"]
           axs[0].plot(time, additional_data[i, :, 1], label=names[i])

    axs[0].set_title("position")
    axs[0].set_ylabel("[m]")
    axs[0].set_xlabel(f"time [s]")
    axs[0].grid()
    axs[0].legend()


    axs[1].plot(time, pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
    axs[1].plot(time, x.detach().cpu().numpy()[:, 2], color="blue", label="true", linestyle="dashed")
    if additional_data != None:
        for i in range(additional_data.size(dim=0)):
           names = ["simulink", "Hub im Regler"]
           axs[1].plot(time, additional_data[i, :, 2], label=names[i])
    axs[1].set_title("speed")
    axs[1].set_ylabel("[m/s]")
    axs[1].set_xlabel(f"time [s]")
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(time, x.detach().cpu().numpy()[:,0], label="pressure")
    if additional_data != None:
       for i in range(additional_data.size(dim=0)):
           names = ["simulink", "Hub im Regler"]
           axs[2].plot(time, additional_data[i, :, 0], label=names[i])
    axs[2].set_title("pressure")
    axs[2].set_ylabel("[Pa]")
    axs[2].set_xlabel(f"time [s]")
    axs[2].grid()
    axs[2].legend()

    plt.grid(True)
    plt.legend()
    plt.show()



def test(data, model, model_type = "or_lstm", window_size=10, display_plots=False, num_of_inits = 5, set_rand_seed=True, physics_rescaling = 0, additional_data=None):
    
    modeltypes = ["lstm_or_nextstep", "mlp_or_nextstep", "tcn_or_nextstep", 
                          "lstm_no_or_nextstep", "mlp_no_or_nextstep", "tcn_no_or_nextstep",
                          "lstm_derivative", "mlp_derivative", "tcn_derivative",
                          "or_lstm", "lstm", "mlp", "gru", "tcn", "or_tcn", "neural_cde", "or_mlp"]
    
    modeltypes_keys ={ "lstm_or_nextstep" : "OR-LSTM", 
                    "mlp_or_nextstep" : "OR-MLP", 
                    "tcn_or_nextstep" : "OR-TCN", 
                    "lstm_no_or_nextstep" : "LSTM", 
                    "mlp_no_or_nextstep" : "MLP",
                    "tcn_no_or_nextstep" : "TCN",
                    "lstm_derivative" : "LSTM-dt",
                    "mlp_derivative" : "MLP-dt",
                    "tcn_derivative" : "TCN-dt",            
                    "or_lstm" : "OR-LSTM-dt",
                    "or_tcn": "OR-TCN-dt", 
                    "or_mlp": "OR-MLP-dt"
                }
   
    if model_type not in modeltypes:
        
        print("Error: model_type = ", model_type, "available options are: ", modeltypes)
        return 0
   
    device = "cpu" if data.get_device() == -1 else "cuda:0"
    
    if data.dim() != 3:
        print("data tensor has unexpected dimension", data.dim(), "expected", 3 )
        return 0
    
    timesteps = data.size(dim=1)

    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0
    total_loss = 0
    total_firsthalf = 0
    total_secondhalf = 0

    np.random.seed(1234)
    test_inits = data.size(dim=0)
    ids = np.random.choice(test_inits, min([50, test_inits]), replace=False)
    ids = np.unique(ids)
    data = data[ids,:, :]

    if set_rand_seed:
     np.random.seed(1234)

    # Type 1: OR derivative prediction 
    if model_type in ["or_lstm", "lstm_derivative"]:
        for i, x in enumerate(data):

            x=x.to(device)        
            x = x.view(1,x.size(dim=0), x.size(dim=1))

    
            with torch.inference_mode():
    
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                
                out, _ = model(x)
                pred[window_size:,1:] = out

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                total_firsthalf += loss_fn(pred[window_size:int((timesteps-window_size)/2), 1:], 
                        x[0, window_size:int((timesteps-window_size)/2), 1:]).detach().cpu().numpy()  
                total_secondhalf += loss_fn(pred[int((timesteps-window_size)/2):, 1:],
                            x[0, int((timesteps-window_size)/2):, 1:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling, additional_data=additional_data)

    if model_type in ["or_mlp", "mlp_derivative"] :
         for i, x in enumerate(data):
            

            with torch.inference_mode():
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                x_test = x.clone()
                x_test[:,window_size:,1:] = 0
                x_test = x_test.to(device)
                #print("Data passed to the model, all 0 after the initial window to prove that the forward pass is correct and doesnt access information it shouldnt.",x_test[:,0:10,:])

                out = model(x_test)
                
                pred[window_size:,1:] = out

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                total_firsthalf += loss_fn(pred[window_size:int((pred.size(dim=0)-window_size)/2), 1:], 
                                            x[0, window_size:int((pred.size(dim=0)-window_size)/2), 1:]).detach().cpu().numpy()  
                total_secondhalf += loss_fn(pred[int((pred.size(dim=0)-window_size)/2):, 1:],
                                                x[0, int((pred.size(dim=0)-window_size)/2):, 1:]).detach().cpu().numpy()  

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling , additional_data=additional_data)

    if model_type in ["or_tcn", "tcn_derivative"] :
         for i, x in enumerate(data):


            with torch.inference_mode():
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                x_test = x.clone()
                x_test[:,window_size:,1:] = 0
                x_test = x_test.to(device)
                #print("Data passed to the model, all 0 after the initial window to prove that the forward pass is correct and doesnt access information it shouldnt.",x_test[:,0:10,:])

                out = model(x_test.transpose(1,2))
                
                pred[window_size:,1:] = out.squeeze(0).transpose(0,1)

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling , additional_data=additional_data)


    # Type 2: next step prediction 
    if model_type in ["lstm_or_nextstep", "lstm_no_or_nextstep"]:
        for i, x in enumerate(data):

            x=x.to(device)        
            x = x.view(1,x.size(dim=0), x.size(dim=1))
   
            with torch.inference_mode():
    
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                
                out, _ = model(x)
                pred[window_size:,1:] = out

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                total_firsthalf += loss_fn(pred[window_size:int((timesteps-window_size)/2), 1:], 
                        x[0, window_size:int((timesteps-window_size)/2), 1:]).detach().cpu().numpy()  
                total_secondhalf += loss_fn(pred[int((timesteps-window_size)/2):, 1:],
                            x[0, int((timesteps-window_size)/2):, 1:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling, additional_data=additional_data)

    if model_type in ["mlp_or_nextstep", "mlp_no_or_nextstep"] :
         for i, x in enumerate(data):

            with torch.inference_mode():
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                x_test = x.clone()
                x_test[:,window_size:,1:] = 0
                x_test = x_test.to(device)
                #print("Data passed to the model, all 0 after the initial window to prove that the forward pass is correct and doesnt access information it shouldnt.",x_test[:,0:10,:])

                out = model(x_test)
                
                pred[window_size:,1:] = out

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                total_firsthalf += loss_fn(pred[window_size:int((pred.size(dim=0)-window_size)/2), 1:], 
                                            x[0, window_size:int((pred.size(dim=0)-window_size)/2), 1:]).detach().cpu().numpy()  
                total_secondhalf += loss_fn(pred[int((pred.size(dim=0)-window_size)/2):, 1:],
                                                x[0, int((pred.size(dim=0)-window_size)/2):, 1:]).detach().cpu().numpy()  

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling , additional_data=additional_data)

    if model_type in ["tcn_or_nextstep", "tcn_no_or_nextstep"] :
         for i, x in enumerate(data):
            
            with torch.inference_mode():
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 3), device=device)
    
                if window_size > 1:
                    pred[0:window_size, :] = x[0, 0:window_size, :]
                    pred[:, 0] = x[0, :, 0]
    
                else:
                    pred[0, :] = x[0, 0, :]
                    pred[:, 0] = x[0, :, 0]
    
                x_test = x.clone()
                x_test[:,window_size:,1:] = 0
                x_test = x_test.to(device)
                #print("Data passed to the model, all 0 after the initial window to prove that the forward pass is correct and doesnt access information it shouldnt.",x_test[:,0:10,:])

                out = model(x_test.transpose(1,2))
                
                pred[window_size:,1:] = out.squeeze(0).transpose(0,1)

                test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, pred_next_step=None, physics_rescaling=physics_rescaling , additional_data=additional_data)
   

    if model_type == "neural_cde" :
         for i, x in enumerate(data):
        

            with torch.inference_mode():

                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))

                pred = torch.zeros_like(x, device=device)
        
                pred[:, 0:window_size, :] = x[0:1, 0:window_size, :]
                pred[:, :, 0:2] = x[0:1, :, 0:2] # time, pressure

                #start_total=time.time()

                for i in range(timesteps - window_size):
                    
                    #start_coeffs=time.time()
                    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(pred[0:1, i:i+window_size, :]) 
                    #train_coeffs = torchcde.linear_interpolation_coeffs(pred[0:1, i:i+window_size, :])
   
                    #stop_coeffs=time.time()
                    #print(stop_coeffs-start_coeffs, "time: coeff calc one step")
                    if (i+1)%100==0:
                     print(i, " timessteps done")
                    #start=time.time()

                    out = model(train_coeffs)
                    pred[0:1, i+window_size, 2:] = pred[0:1, i+window_size-1, 2:] + out.unsqueeze(1)

                    #pred[0:1, i+window_size, 2:] = out
                    #stop=time.time()
                    #print(stop-start, "time: model calc step")

                test_loss += loss_fn(pred[0, window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                test_loss_deriv += loss_fn(pred[0, window_size:, 3], x[0, window_size:, 3]).detach().cpu().numpy()
                total_loss += loss_fn(pred[0, window_size:, 2:], x[0, window_size:, 2:]).detach().cpu().numpy()

                total_firsthalf += loss_fn(pred[0, window_size:int((pred.size(dim=1)-window_size)/2), 2:], 
                                           x[0, window_size:int((pred.size(dim=1)-window_size)/2), 2:]).detach().cpu().numpy()  
                total_secondhalf += loss_fn(pred[0, int((pred.size(dim=1)-window_size)/2):, 2:],
                                             x[0, int((pred.size(dim=1)-window_size)/2):, 2:]).detach().cpu().numpy()  
                #print("Error first half: ", total_firsthalf)
                #print("Error second half: ", total_secondhalf)

                #stop_total=time.time()
               # print(stop_total-start_total, "time: model calc step")

                if display_plots:
                    plot_results(x[:,:,1:], pred[:,:,1:], pred_next_step=None, physics_rescaling=physics_rescaling , additional_data=additional_data)


    return pred, total_loss