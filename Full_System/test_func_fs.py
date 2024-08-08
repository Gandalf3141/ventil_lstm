import torch
from get_data_fs import *
from NN_classes_fs import *
from dataloader_fs import *
from test_func_fs import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def plot_results(x, pred, rescale=False):

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    if pred.dim() == 3:
        pred = pred.view(pred.size(dim=1), pred.size(dim=2))

    if rescale:
        x = normalize_invert(x)
        pred = normalize_invert(pred)

    figure , axs = plt.subplots(5,1, figsize=(9,9))
    figure.tight_layout(pad=2.0)

    stepsize = 2e-5
    time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

    axs[0].plot(time, x.detach().cpu().numpy()[:, 0], color="blue", label="data")
    axs[0].set_title("NC : Input Voltage 1")
    axs[0].set_ylabel("[V]")

    axs[1].plot(time, x.detach().cpu().numpy()[:, 1], color="blue", label="data")
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

    axs[4].plot(time, pred.detach().cpu().numpy()[:, 4], color="red", label="pred")
    axs[4].plot(time, x.detach().cpu().numpy()[:, 4], color="blue", label="data", linestyle="dashed")
    axs[4].set_title("velocity")
    axs[4].set_ylabel("[m/s]")
    axs[4].set_xlabel(f"time [s]")

    for i in range(5):
        axs[i].grid(True)
        axs[i].legend()



    plt.grid(True)
    plt.legend()
    plt.show()

def test(data, model, model_type="tcn", window_size=1 ,display_plots=False, numb_of_inits=1, fix_random=True):

    if fix_random:
     np.random.seed(1234)
 
    test_inits = data.size(dim=0)
    ids = np.random.choice(test_inits, min([numb_of_inits, test_inits]), replace=False)
    ids = np.unique(ids)
    data = data[ids,:, :]

    loss_fn = nn.MSELoss()
    timesteps = data.size(dim=1)

    for i, x in enumerate(data):

        total_loss = 0
        with torch.inference_mode():
            if model_type == "tcn":
                
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 5), device=device)

                pred[0:window_size, :] = x[0, 0:window_size, :]
                pred[:, 0] = x[0, :, 0]

                x_test = x.clone()
                x_test[:,window_size:,2:] = 0
                x_test = x_test.to(device)
                #print("Data passed to the model, all 0 after the initial window to prove that the forward pass is correct and doesnt access information it shouldnt.",x_test[:,0:10,:])

                out = model(x_test.transpose(1,2))
                
                pred[window_size:,2:] = out.squeeze(0).transpose(0,1)

                total_loss += loss_fn(pred[window_size:, 2:], x[0, window_size:, 2:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, rescale=False)

            if model_type == "lstm":
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 5), device=device)

                pred[0:window_size, :] = x[0, 0:window_size, :]
                pred[:, 0] = x[0, :, 0]

                x_test = x.clone()
                x_test[:,window_size:,2:] = 0
                x_test = x_test.to(device)

                out, _ = model(x_test) 
                pred[window_size:,2:] = out

                total_loss += loss_fn(pred[window_size:, 2:], x[0, window_size:, 2:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, rescale=False)

            if model_type == "mlp":
                x=x.to(device)        
                x = x.view(1,x.size(dim=0), x.size(dim=1))                
                pred = torch.zeros((timesteps, 5), device=device)

                pred[0:window_size, :] = x[0, 0:window_size, :]
                pred[:, 0] = x[0, :, 0]
    
                x_test = x.clone()
                x_test[:,window_size:,2:] = 0
                x_test = x_test.to(device)

                out = model(x_test)
                pred[window_size:,2:] = out

                total_loss += loss_fn(pred[window_size:, 2:], x[0, window_size:, 2:]).detach().cpu().numpy()

                if display_plots:
                    plot_results(x, pred, rescale=False)

    return total_loss