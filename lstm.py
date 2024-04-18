# Importing necessary libraries
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
from icecream import ic
from tqdm import tqdm
from itertools import chain
from get_data import get_data
import logging
import os

# Define the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device = "cpu"

class LSTMmodel(nn.Module):
    """
    LSTM model class for derivative estimation.
    """

    def __init__(self, input_size, hidden_size, out_size, layers):
        """
        Initialize the LSTM model.

        Args:
        - input_size: Size of input
        - hidden_size: Size of hidden layer
        - out_size: Size of output
        - layers: Number of layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):
        """
        Forward pass through the LSTM model.

        Args:
        - seq: Input sequence

        Returns:
        - pred: Model prediction
        - hidden: Hidden state
        """
        lstm_out, hidden = self.lstm(seq)
        pred = self.linear(lstm_out.view(len(seq), -1))

        return pred, hidden


def slice_batch(batch, window_size=1):
    """
    Slice the input data into batches for training.

    Args:
    - batch: Input data batch
    - window_size: Size of the sliding window

    Returns:
    - List of sliced batches
    """
    l = []
    for i in range(len(batch) - window_size):
        l.append((batch[i:i+window_size, :], batch[i+1:i+window_size+1, 1:]))
    return l


def train(input_data, model, ws=1):
    """
    Train the LSTM model using input data.

    Args:
    - input_data: Input data for training
    - model: LSTM model to be trained
    - ws: Window size
    - odestep: Option for using ODE steps
    - use_autograd: Option for using autograd

    Returns:
    - Mean loss over all batches
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    total_loss = []

    for batch in input_data:

        input = slice_batch(batch, ws)
        batch_loss = 0

        for inp, label in input:  # inp = (u, x) label = x

            output, _ = model(inp)

            out = inp[:, 1:] + output

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            batch_loss += loss.detach().numpy()

        total_loss.append(batch_loss/len(batch))

    return np.mean(total_loss)


def test(test_data, time, model, plot_opt=False, ws=1, steps=200, odestep=False, use_autograd=False):
    """
    Test the trained LSTM model using test data.

    Args:
    - test_data: Test data
    - time: Time data
    - model: Trained LSTM model
    - plot_opt: Option for plotting results
    - ws: Window size
    - steps: Number of steps
    - odestep: Option for using ODE steps
    - use_autograd: Option for using autograd

    Returns:
    - Mean test loss
    - Mean derivative test loss
    """
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    test_loss_deriv = 0

    for x in test_data:
        with torch.inference_mode():

            pred = torch.zeros((steps, 3))
            pred_next_step = torch.zeros((steps, 3))

            if ws > 1:
                pred[0:ws, :] = x[0:ws, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0:ws, :] = x[0:ws, :]
                pred_next_step[:, 0] = x[:, 0]
            else:
                pred[0, :] = x[0, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0, :] = x[0, :]
                pred_next_step[:, 0] = x[:, 0]

            for i in range(len(x) - ws):
                out, _ = model(pred[i:i+ws, :])

                if odestep:
                    pred[i+ws, 1:] = pred[i+ws-1, 1:] + out[-1, :]
                    pred_next_step[i+ws, 1:] = x[i+ws-1, 1:] + out[-1, :]
                else:
                    pred[i+ws, 1:] = out[-1]
                    outt, _ = model(x[i:i+ws, :])
                    pred_next_step[i+ws, 1:] = outt[-1]

                if use_autograd:
                    print("not implemented yet")

            test_loss += loss_fn(pred[:, 1], x[:, 1]).detach().numpy()
            test_loss_deriv += loss_fn(pred[:, 2], x[:, 2]).detach().numpy()

            if plot_opt:
                plt.plot(time, pred.detach().numpy()[
                         :, 1], color="red", label="pred")
                plt.plot(time, pred_next_step.detach().numpy()[
                         :, 1], color="green", label="next step from data")
                plt.plot(time, x.detach().numpy()[
                         :, 1], color="blue", label="true", linestyle="dashed")

                plt.grid()
                plt.legend()
                plt.show()

    return np.mean(test_loss), np.mean(test_loss_deriv)


def main():

    log_file = 'training.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'

    # Configure logging
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define parameters
    window_size = 10

    losses = []

    # Generate input data
    input_data = get_data(path="save_data_test.csv")

    # Split data into train and test sets
    train_size = int(0.9 * len(input_data))
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])

    # Take a slice of data for training
    slice_of_data = 100

    train_dataset = train_dataset[:][:, 0:slice_of_data, :]

    # Initialize the LSTM model
    model = LSTMmodel(input_size=3, hidden_size=5, out_size=2, layers=1).to(device)

    trained=False
    if trained:
     path = "trained_NNs\lstm_ws4.pth"
     
     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model = LSTMmodel(input_size=3, hidden_size=5, out_size=2, layers=1).to(device)
    
    epochs = 3

    for e in tqdm(range(epochs)):
        loss_epoch = train(train_dataset, model, ws=window_size)
        losses.append(loss_epoch)
        if e % 2 == 0:
            print(f"Epoch {e}: Loss: {loss_epoch}")
    # Plot losses
    plt.plot(losses[1:])
    plt.show()

    # Save trained model
    torch.save(model.state_dict(), "trained_NNs/lstm_ws4.pth")

    # Log parameters
    logging.info(f"Epochs: {epochs}, Window Size: {window_size},\n Start Time: {start_time}, Stop Time: {stop_time}, Timesteps: {timesteps}, Number of Inits: {num_of_inits},\n Option for ODE Step: {option_odestep}")
    logging.info(f"final loss {losses[-1]}")
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")
    logging.info("\n")

    return None


if __name__ == "__main__":
    main()

# chat gpt instructions
# Add short and precise comments to the following python code. Describe functions, classes, loops similar things. The code:
