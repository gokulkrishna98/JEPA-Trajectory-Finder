import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr

from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from lightly.models.modules.heads import VICRegProjectionHead
from encoder_train import save_model, compute_mean_and_std, get_byol_transforms, get_encoder_loss
from encoder_train import criterion as VICReg_criterion
from tqdm import tqdm
from j_cnn_linear import JEPAModelv2

import numpy as np
import math
import matplotlib.pyplot as plt

## setting variables
embed_dim = 1024
epochs = 10
learning_rate = 0.001
use_expander = False
batch_size = 16

dataset_directory = "../dataset"
states_filename = "states.npy"
actions_filename = "actions.npy"

def train_joint(model, dataloader, criterion_encoder, criterion_pred, 
                optimizer, transformation1, transformation2, 
                device, epochs=10, use_expander=False):
    model.to(device)
    model.train()
    
    # clipping the gradient to handle gradient explosions in LSTM
    # max_val = 5.0
    # for param in model.predictor.parameters():
    #     if param.grad is not None:
    #         param.grad.data = torch.clamp(param.grad.data, -max_val, max_val)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc="Processing Batch"):
            state, action = batch
            state, action = state.to(device), action.to(device)
            B, L, D = state.shape[0], action.shape[1], model.predictor.hidden_dim

            loss, loss1, loss2, loss3 = 0, 0, 0, 0

            o = state[:, 0, :, :, :]
            model.set_init_embedding(o, use_expander)

            # compute loss1
            loss1 = get_encoder_loss(model, o, transformation1, transformation2, 
                                     criterion_encoder)
            for i in range(L):
                # inference of encoder(next state) and predictor(action) 
                sy_hat, (sy_enc, sy_exp) = model(action[:, i, :], state[:, i+1, :, :, :])
                sy = sy_exp if use_expander else sy_enc

                # compute loss2 (distance btw sy and sy_hat)
                loss2 += criterion_pred(sy_hat, sy)
                # vic_reg loss for encoder (for encoding next state)
                loss3 += get_encoder_loss(model, state[:, i, :, :, :], 
                                          transformation1, transformation2, 
                                          criterion_encoder) 
            
            # adding all loss and doing back propagation
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch: {epoch}, total_loss: {total_loss}, the avg loss = {total_loss/len(dataloader)}")
        save_model(model, epoch, file_name="join_model")

    return model        


def main():
    ## Loading Dataset
    print("Loading dataset and data loader ....")
    dataset = TrajectoryDataset(
        data_dir = dataset_directory,
        states_filename = states_filename,
        actions_filename = actions_filename,
        s_transform = None,
        a_transform = None,
        length = 1000 
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    first_datapoint = next(iter(dataloader))
    state, action = first_datapoint
    print(f"Number of data_points {len(dataloader)}")
    print(f"Shape of state: {state.shape}")
    print(f"Shape of action: {action.shape}")

    mean, std = compute_mean_and_std(dataloader, is_channelsize3=False)
    transformation1, transformation2 = get_byol_transforms(mean, std)

    print("Defining Model.....")
    model = JEPAModelv2(1024, 4096, 2, 2)

    joint_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1.5e-4)
    # joint_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_predictor = nn.MSELoss()
    criterion_encoder = VICReg_criterion
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_joint(model, dataloader, criterion_encoder, criterion_predictor, 
                joint_optimizer, transformation1, transformation2, device,
                epochs=epochs, use_expander=use_expander)

if __name__ == "__main__":
    main()