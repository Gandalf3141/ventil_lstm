import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataloader_fs import *
from meas_test_func_fs import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def plot_results(x, pred, rescale=False, window_size=1, sim_data_index=0):
    
    if rescale:
        x = normalize_invert(x)
        pred = normalize_invert(pred)

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    if pred.dim() == 3:
        pred = pred.view(pred.size(dim=1), pred.size(dim=2))



    figure , axs = plt.subplots(4,1, figsize=(9,9))
    figure.tight_layout(pad=2.0)

    stepsize = 2 * 2e-5 
    time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

    axs[0].plot(time, x.detach().cpu().numpy()[:, 0], color="darkgreen", label="data")
    axs[0].set_title("NC : Input Voltage 1")
    axs[0].set_ylabel("[V]")

    axs[1].plot(time, x.detach().cpu().numpy()[:, 1], color="darkgreen", label="data")
    axs[1].set_title("NO : Input Voltage 2")
    axs[1].set_ylabel("[V]")

    axs[2].plot(time, pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
    axs[2].plot(time, x.detach().cpu().numpy()[:, 2], color="blue", label="data", linestyle="dashed")
    axs[2].set_title("pressure")
    axs[2].set_ylabel("[Pa]")

    axs[3].plot(time, pred.detach().cpu().numpy()[:, 3], color="red", label="pred")
    axs[3].plot(time, x.detach().cpu().numpy()[:, 3], color="blue", label="data", linestyle="dashed")
    axs[3].set_title("position")
    axs[3].set_ylabel("[m]")
    axs[3].set_xlabel(f"time [s]")

    axs[2].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')
    axs[3].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')


    if sim_data_index >= 0:
        path_test_data_simulink = r"C:\Users\strasserp\Documents\ventil_lstm\Experiment_Meassurements\Messungen\Messdaten_Simulink_Vergleich.csv"
        test_data_simulink = get_data(path_test_data_simulink, num_inits=0)
        #test_data_simulink[id,:,2]
        axs[2].plot(time, test_data_simulink[sim_data_index,:,2], color="orange", label="pressure_simulink")
        axs[3].plot(time, test_data_simulink[sim_data_index,:,3], color="orange", label="position_simulink")


    if rescale:
        u1_max = 200  #Spannung in [V]              ... [0, 200]
        u1_min = 0
        u2_max = 200
        u2_min = 0
        p_max = 3.5*1e5 #Druck in [bar]             ... [1, 3.5]
        p_min = 1.0*1e5 #Umgebungsdruck in [bar]
        s_max = 0.605*1e-3     #Position [m]          ... [0, 0.0006]
        s_min = 0.0

        axs[0].set_ylim(u1_min-10, u1_max+10)
        axs[1].set_ylim(u2_min-10, u2_max+10)
        axs[2].set_ylim(p_min-0.1*p_max, p_max+0.1*p_max)
        axs[3].set_ylim(s_min-+0.1*s_max, s_max+0.1*s_max)

    for i in range(4):
        axs[i].grid(True)
        axs[i].legend()



    plt.grid(True)
    plt.legend()
    plt.show()

def test(data, model, model_type="lstm", window_size=1 ,display_plots=False, numb_of_inits=1, fix_random=True, rescale=False, specific_index=-1):

    if fix_random:
     np.random.seed(1234)
    else:
     np.random.seed(seed=None)
      
    test_inits = data.size(dim=0)
    ids = np.random.choice(test_inits, min([numb_of_inits, test_inits]), replace=False)
    ids = np.unique(ids)

    

    if specific_index >= 0:
        data = data[specific_index:specific_index+1,:, :]
    else :
        data = data[ids,:, :]   
    loss_fn = nn.MSELoss()
    timesteps = data.size(dim=1)

    total_loss = 0
    for i, x in enumerate(data):

        with torch.inference_mode():
            if model_type == "lstm":
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 4), device=device)

                pred[0:window_size, :] = x[0, 0:window_size, :]
                pred[:, 0] = x[0, :, 0]

                x_test = x.clone()
                x_test[:,window_size:,2:] = 0
                x_test = x_test.to(device)

                out, _ = model(x_test) 
                pred[window_size:,2:] = out

                total_loss += loss_fn(pred[window_size:, 2:], x[0, window_size:, 2:]).detach().cpu().numpy()

                if display_plots:
                    if specific_index>=0:
                        plot_results(x, pred, rescale=rescale, window_size=window_size, sim_data_index=specific_index)
                    else:
                        plot_results(x, pred, rescale=rescale, window_size=window_size, sim_data_index=ids[i])

    return total_loss/data.size(0)