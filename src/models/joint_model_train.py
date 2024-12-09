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



## Model definition
class SimpleEncoder(nn.Module):
    def __init__(self, embed_size, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 12, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((5, 5), stride=2)
        self.pool2 = nn.MaxPool2d((5, 5), stride=5)
        self.fc1 = nn.Linear(432, 4096)
        self.fc2 = nn.Linear(4096, embed_size)

    def forward(self, x):
        # h,w = 65
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x1
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x3 + x2
        x3 = self.pool2(x3)

        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        return x3

class VICRegModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=1024,
            hidden_dim=2048,
            output_dim=1024,
            num_layers=3,
        )
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.h = None
        self.c = None

    def set_hc(self, h, c):
        self.h = h
        self.c = c 
    
    def reset_hc(self):
        self.h = self.h.zero_() 
        self.c = self.c.zero_()

    def forward(self, action):
        self.h, self.c = self.lstm_cell(action, (self.h, self.c))
        return self.h

class JEPAModel(nn.Module):
    def __init__(self, embed_size, input_channel_size):
        super().__init__()
        self.encoder = VICRegModel(SimpleEncoder(embed_size, input_channel_size))
        self.predictor = Predictor(input_channel_size, embed_dim)
        
    def set_predictor(self, o, co, use_expander=False):
        x, z = self.encoder.forward(o)
        so = z if use_expander else x
        self.predictor.set_hc(so, co)
        return so
    
    def reset_predictor(self):
        self.predictor.reset_hc()

    def forward(self, action=None, state=None):
        sy_hat, sy = None, None
        if action is not None:
            sy_hat = self.predictor(action)
        if state is not None:
            sy = self.encoder(state)

        return sy_hat, sy

    def forward_inference(self, actions, states):
        B, L, D = states.shape[0], actions.shape[1], self.predictor.hidden_size

        o = states[:, 0, :, :, :]
        co = torch.zeros((B, D)).to(o.device)
        self.set_predictor(o, co, use_expander=False)

        result = torch.empty((B, L, D))
        for i in range(L):
            sy_hat, _ = self.forward(actions[:, i, :], state=None)
            result[:, i, :] = sy_hat

        return result



def train_joint(model, dataloader, criterion_encoder, criterion_pred, 
                optimizer, transformation1, transformation2, 
                device, epochs=10, use_expander=False):
    model.to(device)
    model.train()
    
    # clipping the gradient to handle gradient explosions in LSTM
    max_val = 5.0
    for param in model.predictor.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -max_val, max_val)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc="Processing Batch"):
            state, action = batch
            state, action = state.to(device), action.to(device)
            B, L, D = state.shape[0], action.shape[1], model.predictor.hidden_size

            loss, loss1, loss2, loss3 = 0, 0, 0, 0

            o = state[:, 0, :, :, :]
            c0 = torch.zeros((B, D)).to(device)
            model.set_predictor(o, c0, use_expander)

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
    model = JEPAModel(embed_dim, 2)

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