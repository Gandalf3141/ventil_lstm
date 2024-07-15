
experiment set up:

Training:

> These things stay constant: epochs, batchsize, ~model size, 
> save NN(test_set) every 10% epochs during training

Train all networks (LSTM, MLP, TCN) with 4 Settings:

# Type 1: OR derivative prediction 
# Type 2: next step prediction 
# Type 3: derivative prediction
# Type 4: next step prediction 

!!!
ALL experiments are performed on these (fixed) networks 
!!!

---------------------------------------------------------------------------------------

1. experiment:

Compare the performance of the choosen network architectures

Train and run test set every 10% of epochs (save for train plots)

##################################################################

2. experiment:

compare x_k+1 = NN(x_k) and x_k+1 = x_k + NN(x_k)

Note:

choose data that covers only part of the possible inputs (u(t))

##################################################################

3. experiment:

compare OR and TF in Training.

Note:

maybe for each network architecture individually?

##################################################################

