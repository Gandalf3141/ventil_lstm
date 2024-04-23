import pandas as pd
import torch

def get_data(path = "ventil_lstm\save_data_test.csv"):
    
    df = pd.read_csv(path, header=0, nrows=600, skiprows=[x for x in range(1,100)])

    #Reorder columns for familiar setup (t,u,x) here (t, p_b, s_b, w_b)
    L = df.columns.to_list()
    time_cols = L[0::4]
    sb_cols = L[1::4]
    pb_cols = L[2::4]
    wb_cols = L[3::4]
    new_col_order = [x for sub in list(zip(time_cols, pb_cols, sb_cols, wb_cols)) for x in sub]
    df= df[new_col_order]
    df = df.drop(time_cols, axis=1)

    #normalise each column of the dataframe
    #mean normalization
    #df=(df-df.mean())/df.std()

    #min max normalization
    #normalize only a part of the data(??)
    df[sb_cols+wb_cols]=(df[sb_cols+wb_cols]-df[sb_cols+wb_cols].min())/(df[sb_cols+wb_cols].max()-df[sb_cols+wb_cols].min())
    
    #Can't normalize p_b because then a[i]*X+b[i] becomes cX+d for all i.. same with mean normal. 
    
    df[pb_cols] = df[pb_cols] / 1e5

    tensor = torch.tensor(df.values)

    #tensor with t=0:600, 500 different input and the 4 outputs [time, s_b, p_b, w_b]
    tensor = tensor.view(600,500,3).permute(1,0,2)


    return tensor
