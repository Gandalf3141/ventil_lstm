import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_data(path = "ventil_lstm\save_data_test.csv", timesteps_from_data=100, skip_steps_start = 1, skip_steps_end = 1, drop_half_timesteps = True, normalise_s_w="mean", rescale_p=False, num_inits=0):
    
    if path[-3:] == "pkl":
       
     df = pd.read_pickle(path)

    else:
       if timesteps_from_data>1:
        df = pd.read_csv(path, header=0, nrows=timesteps_from_data, skiprows=skip_steps_start)
       else:
        df = pd.read_csv(path, header=0, skiprows=skip_steps_start)


    #drop even more timesteps
    if drop_half_timesteps:
     df = df.iloc[::2]

    if skip_steps_end>1:
       df = df.iloc[0:len(df)-skip_steps_end]

    if num_inits>1:
       df = df.iloc[:,0:4*num_inits]

    #Reorder columns for familiar setup (t,u,x) here (t, p_b, s_b, w_b)
    L = df.columns.to_list()
    time_cols = L[0::4]
    sb_cols = L[1::4]
    pb_cols = L[2::4]
    wb_cols = L[3::4]
    new_col_order = [x for sub in list(zip(time_cols, pb_cols, sb_cols, wb_cols)) for x in sub]
    df= df[new_col_order]
    df = df.drop(time_cols, axis=1)

    #
    PSW_max = 0
    # Normalise / Rescale
    if normalise_s_w=="mean":
        tmp=pb_cols+sb_cols+wb_cols
        #mean normalization
        df[tmp]=(df[tmp]-df[tmp].mean())/df[tmp].std()
        #min-max normalization
        #df[tmp]=(df[tmp]-df[tmp].min())/(df[tmp].max()-df[tmp].min())  
    elif normalise_s_w == "minmax":

        p_max = 3.5*1e5 #Druck in [bar]             ... [0, 3.5]
        s_max = 0.6*1e-3 #Position [m]          ... [0, 0.0006]
        w_max = 1.7 #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]

        p_min = 1.0
        s_min = 0.0
        w_min = -1.7

        # alt 
        # p_max = np.max(df[pb_cols].to_numpy())
        # p_min = np.min(df[pb_cols].to_numpy())

        # s_max = np.max(df[sb_cols].to_numpy())
        # s_min = np.min(df[sb_cols].to_numpy())

        # w_max = np.max(df[wb_cols].to_numpy())
        # w_min = np.min(df[wb_cols].to_numpy())
        # PSW_max = [p_max, s_max, w_max, p_min, s_min, w_min]
        # print(PSW_max)

        df[pb_cols]=(df[pb_cols] - p_min) / (p_max - p_min)
        df[sb_cols]=(df[sb_cols] - s_min) / (s_max - s_min)
        df[wb_cols]=(df[wb_cols] - w_min) / (w_max - w_min)     

        #PSW_max = [p_max, s_max, w_max]

    if rescale_p:
       
       df[pb_cols] = df[pb_cols] / 1e5 

    tensor = torch.tensor(df.values)

    #tensor with t=0:600, 500 different input and the 3 outputs [s_b, p_b, w_b]
    a = num_inits if num_inits>0 else 500
    a=int(len(df.columns.to_list())/3)

    tensor = tensor.view(len(df),a,3).permute(1,0,2)



    # Daten aus Matlab Simulation bzw. physikalische Größen
    # Als Ober/Unter Grenzen

    p_max = 3.5*1e5 #Druck in [bar]             ... [1 , 3.5]
    s_max = 0.6*1e-3 #Position [m]          ... [0, 0.0006]
    w_max = 1.7 #Geschwindigkeit in [m/s]   ... [-1.7, 1.7]
    
    #Diese Werte bestätigen!!!
    p_min = 1.0
    s_min = 0.0
    w_min = -1.7

    PSW_max = [p_max, s_max, w_max, p_min, s_min, w_min]


    return tensor, PSW_max


def visualise(data, num_inits=499,set_ylim=False):
 
    steps=data.size(dim=1) 
    
    if num_inits >= data.size(dim=0):
        
       print("index exceeds number of initial conditions -> random value chosen")
       ids = np.random.randint(0,data.size(dim=0),1)
    else:
       ids = [num_inits]
    
    
    figure , axs = plt.subplots(1, 3, figsize=(16,9))
    colors=["r","b","g","yellow", "purple"]
    for k, id in enumerate(ids):
        if set_ylim:
            axs[0].set_ylim(0, 3.6*1e5)
            axs[1].set_ylim(0, 0.7*1e-3)
            axs[2].set_ylim(-1, 1)

        
        axs[0].plot(data[id,:,1], label="position", color=colors[k+1], linewidth=3, alpha=0.7)
        axs[1].plot(data[id,:,2], label="speed", color=colors[k+2], linewidth=3, alpha=0.7)
        axs[2].plot(data[id,:,0], label="pressure", color=colors[k], linewidth=3, alpha=0.7)

        axs[0].grid(True)
        #axs[0].legend()
        axs[2].set_title("pressure [Pa]")
        axs[1].grid(True)
        #axs[1].legend()
        axs[0].set_title("position [m]")
        axs[2].grid(True)
       #axs[2].legend()
        axs[1].set_title("speed [m/s]")

    # ids = np.random.randint(0,400,2)

    # figure2 , axs = plt.subplots(1, len(ids))
    # for j, id in enumerate(ids):
    #     axs[j].plot(np.linspace(0,1,steps), data[id,:,0], label="pressure")
    #     axs[j].plot(np.linspace(0,1,steps), data[id,:,1], label="position")
    #     axs[j].plot(np.linspace(0,1,steps), data[id,:,2], label="speed")
    #     axs[j].grid(True)
    #     axs[j].legend()
    #     axs[j].set_title("Ventil Sim-daten")

    plt.show()

#visualise(get_data(path = "save_data_test3.csv", timesteps_from_data=0, skip_steps_start = 0, skip_steps_end = 0, drop_half_timesteps = False, normalise_s_w=False, rescale_p=True, num_inits=0), num_inits=20)



