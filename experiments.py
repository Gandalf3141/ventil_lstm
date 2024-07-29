# This file contains all the experiments to compare the trained models 
import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt

def exp1(models: dict, data, window_sizes, plot_errs=False, set_random=False):
    
    if set_random:
         np.seed(1234)
    
    if plot_errs==False:
        index = np.random.randint(0, data.size(dim=0),1)[0]
        data = data[index:index+1,:,:]


    
    test_loss = 0
    test_loss_deriv = 0
    total_loss = 0
    total_firsthalf = 0
    total_secondhalf = 0

    device = "cpu" if data.get_device() == -1 else "cuda:0"
    timesteps = data.size(dim=1)
    loss_fn = torch.nn.MSELoss()

    predictionary = {}
    error_dict = {model_type : [] for model_type in list(models.keys())}
    with torch.inference_mode():
        for model_type, model in models.items():

            window_size = window_sizes[model_type]
            model.eval()
                
            if model_type == "or_lstm":
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

                            error_dict[model_type].append(total_loss)

                            total_firsthalf += loss_fn(pred[window_size:int((timesteps-window_size)/2), 1:], 
                                    x[0, window_size:int((timesteps-window_size)/2), 1:]).detach().cpu().numpy()  
                            total_secondhalf += loss_fn(pred[int((timesteps-window_size)/2):, 1:],
                                        x[0, int((timesteps-window_size)/2):, 1:]).detach().cpu().numpy()
                            
                    predictionary[model_type] = pred
                            
            
            if model_type == "mlp":
                for i, x in enumerate(data):

                        x=x.to(device)
                        
                        with torch.inference_mode():
                
                            pred = torch.zeros((timesteps, 3), device=device)
                
                            if window_size > 1:
                                pred[0:window_size, :] = x[0:window_size, :]
                                pred[:, 0] = x[ :, 0]
                
                            else:
                                pred[0, :] = x[0, :]
                                pred[:, 0] = x[:, 0]

                            inp = torch.cat((x[:window_size,0], x[:window_size,1], x[:window_size,2]))

                            for t in range(1,timesteps - window_size + 1 ): 

                                out = model(inp)
                                pred[window_size+(t-1):window_size+t,1:] =  pred[window_size+(t-2):window_size+(t-1):,1:] + out
                                new_p = pred[t:t+window_size,0]
                                new_s = pred[t:t+window_size,1]
                                new_v = pred[t:t+window_size,2]
                                
                                inp = torch.cat((new_p, new_s, new_v))

                            test_loss += loss_fn(pred[window_size:, 1], x[window_size:, 1]).detach().cpu().numpy()
                            test_loss_deriv += loss_fn(pred[window_size:, 2], x[window_size:, 2]).detach().cpu().numpy()
                            total_loss += loss_fn(pred[window_size:, 1:], x[window_size:, 1:]).detach().cpu().numpy()

                            error_dict[model_type].append(total_loss)

                            
                            total_firsthalf += loss_fn(pred[window_size:int((pred.size(dim=0)-window_size)/2), 1:], 
                                                    x[window_size:int((pred.size(dim=0)-window_size)/2), 1:]).detach().cpu().numpy()  
                            total_secondhalf += loss_fn(pred[int((pred.size(dim=0)-window_size)/2):, 1:],
                                                        x[int((pred.size(dim=0)-window_size)/2):, 1:]).detach().cpu().numpy()  
                            
                predictionary[model_type] = pred

            if model_type == "lstm":
                for i, x in enumerate(data):
                    x=x.to(device)
                
                    with torch.inference_mode():

                        pred = torch.zeros((timesteps, 3), device=device)
                        pred_next_step = torch.zeros((timesteps, 3), device=device)

                        if window_size > 1:
                            pred[0:window_size, :] = x[0:window_size, :]
                            pred[:, 0] = x[:, 0]
                            pred_next_step[0:window_size, :] = x[0:window_size, :]
                            pred_next_step[:, 0] = x[:, 0]
                        else:
                            pred[0, :] = x[0, :]
                            pred[:, 0] = x[:, 0]
                            pred_next_step[0, :] = x[0, :]
                            pred_next_step[:, 0] = x[:, 0]

                        for i in range(len(x) - window_size):

                            out, _ = model(pred[i:i+window_size, :])
                            pred[i+window_size, 1:] = pred[i+window_size-1, 1:] + out[-1, :]
                            pred_next_step[i+window_size, 1:] = x[i+window_size-1, 1:] + out[-1, :]
                        
                        test_loss += loss_fn(pred[:, 1], x[:, 1]).detach().cpu().numpy()
                        test_loss_deriv += loss_fn(pred[:, 2], x[:, 2]).detach().cpu().numpy()

                        total_loss += loss_fn(pred[:, 1:], x[:, 1:]).detach().cpu().numpy()
                        error_dict[model_type].append(total_loss)
                    
                predictionary[model_type] = pred

            if model_type == "tcn" :
                    for i, x in enumerate(data):

                        with torch.inference_mode():

                            x=x.to(device)        
                            x = x.view(1,x.size(dim=0), x.size(dim=1))

                            pred = torch.zeros_like(x, device=device)  
                            pred_next_step = torch.zeros_like(x, device=device)               

                            pred[:, 0:window_size, :] = x[0, 0:window_size, :]
                            pred[:, :, 0] = x[0, :, 0]

                            for i in range(1,x.size(1) - window_size + 1):

                                pred[:, window_size+(i-1):window_size+i,1:] =  pred[:, window_size+(i-2):window_size+(i-1):,1:] + model(pred[:,i:window_size+(i-1),:].transpose(1,2))    

                            test_loss += loss_fn(pred[0, :, 1], x[0, :, 1]).detach().cpu().numpy()
                            test_loss_deriv += loss_fn(pred[0, :, 2], x[0, :, 2]).detach().cpu().numpy()

                            total_loss += loss_fn(pred[0, :, 1:], x[0, :, 1:]).detach().cpu().numpy()
                            error_dict[model_type].append(total_loss)

                        predictionary[model_type] = pred

            if model_type == "or_tcn" :
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
                            error_dict[model_type].append(total_loss)

                        predictionary[model_type] = pred

            if model_type == "neural_cde" :
                    for i, x in enumerate(data):

                        with torch.inference_mode():

                            x=x.to(device)        
                            x = x.view(1,x.size(dim=0), x.size(dim=1))

                            pred = torch.zeros_like(x, device=device)
                    
                            pred[:, 0:window_size, :] = x[0:1, 0:window_size, :]
                            pred[:, :, 0:2] = x[0:1, :, 0:2] # time, pressure

                            #start_total=time.time()

                            for i in range(x.size(1) - window_size):
                                
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
                            error_dict[model_type].append(total_loss)

                            total_firsthalf += loss_fn(pred[0, window_size:int((pred.size(dim=1)-window_size)/2), 2:], 
                                                        x[0, window_size:int((pred.size(dim=1)-window_size)/2), 2:]).detach().cpu().numpy()  
                            total_secondhalf += loss_fn(pred[0, int((pred.size(dim=1)-window_size)/2):, 2:],
                                                            x[0, int((pred.size(dim=1)-window_size)/2):, 2:]).detach().cpu().numpy()  
                    
                        predictionary[model_type] = pred

            if model_type == "or_mlp" :
                    
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

                            out = model(x_test)
                            
                            pred[window_size:,1:] = out

                            test_loss += loss_fn(pred[window_size:, 1], x[0, window_size:, 1]).detach().cpu().numpy()
                            test_loss_deriv += loss_fn(pred[window_size:, 2], x[0, window_size:, 2]).detach().cpu().numpy()
                            total_loss += loss_fn(pred[window_size:, 1:], x[0, window_size:, 1:]).detach().cpu().numpy()
                            error_dict[model_type].append(total_loss)

                            total_firsthalf += loss_fn(pred[window_size:int((pred.size(dim=0)-window_size)/2), 1:], 
                                                        x[0, window_size:int((pred.size(dim=0)-window_size)/2), 1:]).detach().cpu().numpy()  
                            total_secondhalf += loss_fn(pred[int((pred.size(dim=0)-window_size)/2):, 1:],
                                                        x[0, int((pred.size(dim=0)-window_size)/2):, 1:]).detach().cpu().numpy()
                            
                        predictionary[model_type] = pred

 # Type 2: next step prediction 
            if model_type == "lstm_or_nextstep":
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
                        error_dict[model_type].append(total_loss)

                        total_firsthalf += loss_fn(pred[window_size:int((timesteps-window_size)/2), 1:], 
                                x[0, window_size:int((timesteps-window_size)/2), 1:]).detach().cpu().numpy()  
                        total_secondhalf += loss_fn(pred[int((timesteps-window_size)/2):, 1:],
                                    x[0, int((timesteps-window_size)/2):, 1:]).detach().cpu().numpy()
                        predictionary[model_type] = pred


            if model_type == "mlp_or_nextstep" :
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
                        error_dict[model_type].append(total_loss)

                        total_firsthalf += loss_fn(pred[window_size:int((pred.size(dim=0)-window_size)/2), 1:], 
                                                    x[0, window_size:int((pred.size(dim=0)-window_size)/2), 1:]).detach().cpu().numpy()  
                        total_secondhalf += loss_fn(pred[int((pred.size(dim=0)-window_size)/2):, 1:],
                                                        x[0, int((pred.size(dim=0)-window_size)/2):, 1:]).detach().cpu().numpy()
                predictionary[model_type] = pred

            if model_type == "tcn_or_nextstep" :
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
                        error_dict[model_type].append(total_loss)
                predictionary[model_type] = pred

        if not plot_errs:
            phase_plot_predictions(predictionary, data)
            plot_predictions(predictionary, data)
        else:
            plot_errors(error_dict)


def plot_predictions(predictionary, x):



    p_max = 3.5*1e5 #Druck in [bar]         ... [1 , 3.5]
    s_max = 0.6*1e-3 #Position [m]          ... [0, 0.0006]
    w_max = 1.7 #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]
    p_min = 1.0
    s_min = 0.0
    w_min = -1.7
    physics_rescaling = [p_max, s_max, w_max, p_min, s_min, w_min]

    colors = {"or_lstm" : "red",
              "lstm_or_nextstep" : "red",
              "or_mlp" : "green",
              "mlp_or_nextstep" : "green",
              "mlp" : "green",
              "or_tcn" : "purple",
              "tcn_or_nextstep" : "purple",
              "neural_cde"  : "brown"}
    
    if "or_lstm" in list(predictionary.keys()) and "lstm_or_nextstep" in list(predictionary.keys()):
        colors = {"or_lstm" : "red", "lstm_or_nextstep" : "yellow", "lstm_no_or_nextstep" : "green", "lstm_derivative" : "purple"}

    figure , axs = plt.subplots(3,1, figsize=(16,9))
    figure.tight_layout(pad=5.0)    

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    
        #scale back:    
    if physics_rescaling != None:

        x[:,0] = x[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
        x[:,1] = x[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
        x[:,2] = x[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

    greek_letterz=[chr(code) for code in range(945,970)]
    mu = greek_letterz[11]

    stepsize = 2e-5
    time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

    #data
    axs[0].plot(time, x.detach().cpu().numpy()[:, 1], color="blue", label="true", linestyle="dashed")
    axs[1].plot(time, x.detach().cpu().numpy()[:, 2], color="blue", label="true", linestyle="dashed")
    axs[2].plot(time, x.detach().cpu().numpy()[:,0], label="pressure")

    #predictions
    for key, pred in predictionary.items():

        if physics_rescaling != None:

            # we invert:
            # x = (x - xmin)/(xmax - xmin)
            # x * (xmax - xmin) + xmin

            pred[:,0] = pred[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
            pred[:,1] = pred[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
            pred[:,2] = pred[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

        if pred.dim() == 3:
            pred = pred.view(pred.size(dim=1), pred.size(dim=2))

        axs[0].plot(time, pred.detach().cpu().numpy()[:, 1], color=colors[key], label=f"{key}-prediciton", alpha=0.5)
        axs[1].plot(time, pred.detach().cpu().numpy()[:, 2], color=colors[key], label=f"{key}-prediciton", alpha=0.5)


    axs[0].set_title("position")
    axs[0].set_ylabel("[m]")
    axs[0].set_xlabel(f"time [s]")
    axs[0].grid()
    axs[0].legend()   
    axs[1].set_title("speed")
    axs[1].set_ylabel("[m/s]")
    axs[1].set_xlabel(f"time [s]")
    axs[1].grid()
    axs[1].legend()
    axs[2].set_title("pressure")
    axs[2].set_ylabel("[Pa]")
    axs[2].set_xlabel(f"time [s]")
    axs[2].grid()
    axs[2].legend()

    plt.grid(True)
    plt.legend()
    plt.show()


def phase_plot_predictions(predictionary, x):



    p_max = 3.5*1e5 #Druck in [bar]         ... [1 , 3.5]
    s_max = 0.6*1e-3 #Position [m]          ... [0, 0.0006]
    w_max = 1.7 #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]
    p_min = 1.0
    s_min = 0.0
    w_min = -1.7
    physics_rescaling = [p_max, s_max, w_max, p_min, s_min, w_min]

    colors = {"or_lstm" : "red",
              "or_mlp" : "green",
              "or_tcn" : "purple",
              "neural_cde"  : "brown"}

    figure , axs = plt.subplots(1,1, figsize=(16,16))
    figure.tight_layout(pad=5.0)    

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    
        #scale back:    
    if physics_rescaling != None:

        x[:,0] = x[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
        x[:,1] = x[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
        x[:,2] = x[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

    greek_letterz=[chr(code) for code in range(945,970)]
    mu = greek_letterz[11]

    stepsize = 2e-5
    time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

    #data
    axs.plot(x.detach().cpu().numpy()[:, 1],x.detach().cpu().numpy()[:, 2], color="blue", label="true", linestyle="dashed")

    #predictions
    for key, pred in predictionary.items():

        if physics_rescaling != None:

            # we invert:
            # x = (x - xmin)/(xmax - xmin)
            # x * (xmax - xmin) + xmin

            pred[:,0] = pred[:,0]*(physics_rescaling[0] - physics_rescaling[3]) + physics_rescaling[3]
            pred[:,1] = pred[:,1]*(physics_rescaling[1] - physics_rescaling[4]) + physics_rescaling[4]
            pred[:,2] = pred[:,2]*(physics_rescaling[2] - physics_rescaling[5]) + physics_rescaling[5]

        if pred.dim() == 3:
            pred = pred.view(pred.size(dim=1), pred.size(dim=2))

        axs.plot(pred.detach().cpu().numpy()[:, 1], pred.detach().cpu().numpy()[:, 2], color=colors[key], label=f"{key}-prediciton", alpha=0.5)

    axs.set_title("position")
    axs.set_ylabel("[m]")
    axs.set_xlabel(f"time [s]")
    axs.grid()
    axs.legend()   


    plt.grid(True)
    plt.legend()
    plt.show()


def plot_errors(predictionary, x):
     
    # Plot histograms with MSE over many trajectories

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each histogram on the same axes
    for key, values in predictionary.items():
        ax.hist(values, bins=10, alpha=0.5, label=key)

    # Set titles and labels
    ax.set_title('Mean squared errors over X trajectories')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')

    # Add a legend
    ax.legend()
    # Display the plot
    plt.show()
    return None