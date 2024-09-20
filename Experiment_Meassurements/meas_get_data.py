import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data for full system.

# Structure : [Batchsize, Sequence Length, Values] (switches for TCN but not here!)
# Values : [(time), u1, u2, p, s] (time is not returned!)
# Normalized : "minmax" is default - data is always in [0, 1]
 
#physical constants
u1_max = 200  #Spannung in [V]              ... [0, 200]
u1_min = 0
u2_max = 200
u2_min = 0
p_max = 3.5#3.5*1e5 #Druck in [bar]             ... [1, 3.5]
p_min = 1.0#1.0*1e5 #Umgebungsdruck in [bar]a

# !!! Position is already between 0 and 1 !!!
s_max = 1#0.605*1e-3     #Position [m]          ... [0, 0.0006]
s_min = 0.0#0.0
s_max_inv = 0.605*1e-3     #Position [m]          ... [0, 0.0006]
s_min_inv = 0.0#0.0
#v_max = 0.6     #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]
#v_min = -0.6    #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]
x_max = torch.tensor([u1_max, u2_max, p_max, s_max])
x_min = torch.tensor([u1_min, u2_min, p_min, s_min])
x_max_inv = torch.tensor([u1_max, u2_max, p_max, s_max_inv])
x_min_inv = torch.tensor([u1_min, u2_min, p_min, s_min_inv])

def normalize(data):

    # This function normalizes any tensor of shape (batchsize, length, 5)
    # maybe check if time is provided 

    data = data.clone()
    data = (data - x_min) / (x_max - x_min)
    return data


def normalize_invert(data):
    
    # This function inverts the minmax normalization back to the original scale.
    data = data.clone()
    data = data * (x_max_inv - x_min_inv) + x_min_inv
    return data


def get_data(path, num_inits=0):

    df = pd.read_csv(path)

    if num_inits>1:
       df = df.iloc[:,0:4*num_inits]

    tensor = torch.tensor(df.values)

    a = num_inits if num_inits>0 else 500
    a=int(len(df.columns.to_list())/4)

    tensor = tensor.view(len(df),a,4).permute(1,0,2)
    tensor = normalize(tensor)

    return tensor


def visualise(data, num_inits):

    stepsize = 2 * 2e-5
    time = np.linspace(0,data.size(dim=1)*stepsize, data.size(dim=1))
    time = torch.tensor(time)

    # if num_inits >= data.size(dim=0):
        
    #    print("index exceeds number of initial conditions -> random value chosen")
    #    ids = np.random.randint(0,data.size(dim=0),1)
    # else:
    ids = [num_inits]    
    
    figure , axs = plt.subplots(4, 1, figsize=(9,9))
    figure.tight_layout(pad=2.0) 
    colors=["r","b","g","orange", "purple"]
    for k, id in enumerate(ids):

        if data[0,10,2] > 2:
            axs[0].set_ylim(u1_min, u1_max)
            axs[1].set_ylim(u2_min, u2_max)
            axs[2].set_ylim(p_min, p_max)
            axs[3].set_ylim(s_min_inv, s_max_inv)
        #else:
           #for i in range(4):
              #axs[i].set_ylim(-0.1,1.1)
        
        axs[0].plot(time, data[id,:,0], label="NC voltage 1", color=colors[0], linewidth=2)
        axs[1].plot(time, data[id,:,1], label="NO voltage 2", color=colors[1], linewidth=2)
        axs[2].plot(time, data[id,:,2], label="pressure", color=colors[2], linewidth=2)
        axs[3].plot(time, data[id,:,3], label="position", color=colors[3], linewidth=2)
        axs[3].set_xlabel(f"time [s]    stepsize = 2e-5")

        for i in range(4):
            axs[i].grid(True)
            axs[i].legend()
        
        #axs[2].set_title("pressure [Pa]")

    plt.show()

