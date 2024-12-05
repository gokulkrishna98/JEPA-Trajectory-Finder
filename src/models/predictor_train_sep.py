import numpy as np
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tr
import torch.optim as optim
from tqdm import tqdm

from dataset import TrajectoryDataset
from predictor import Predictor
from encoder import Encoder

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def save_model(model, epoch, save_path="checkpoints_pred", file_name="pred"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


def train_predictor(pred, enc, dataloader, criterion, optimizer, device, epochs=10):
    # keeping encoder in eval mode
    pred, enc = pred.to(device), enc.to(device)
    enc.eval()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc="Processing batch"):
            ## shape of [ s = (b, L+1, c, h, w)]  [a = (b, L, 2)]
            s, a = batch
            s, a = s.to(device), a.to(device)

            ## initial observation
            o = s[:, 0, :, :, :]
            o = torch.cat([o, o[:, 1:2, :, :]], dim=1)
            
            with torch.no_grad():  # No gradients for encoder
                so = enc(o)

            co = torch.zeros(so.shape).to(device)
            pred.set_hc(so, co)
            
            loss ,L = 0, a.shape[1]
            for i in range(L):
                sy_hat = pred(a[:, i, :])
                temp = s[:, i+1, :, :, :]
                temp = torch.cat([temp, temp[:, 1:2, :, :]], dim=1)
                
                with torch.no_grad():  # No gradients for encoder
                    sy = enc(temp)
                
                loss += criterion(sy_hat, sy)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## clearing the hidden state and cell state
            pred.reset_hc()
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        save_model(pred, epoch)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.9f}")
    print("Training completed..")
    return pred


if __name__ == "__main__":
    print("Training main function")
    device = get_device()

    dataset = TrajectoryDataset(
        data_dir = "../dataset",
        states_filename = "states.npy",
        actions_filename = "actions.npy",
        s_transform = None,
        a_transform = None,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Dataloader successful")

    # Resnet Encoder
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    encoder = Encoder(backbone).to(device)

    input_size = 2
    hidden_size = 1024
    encoder.load_state_dict(torch.load("enoder_encoder_final.pth"))

    predictor = Predictor(input_size=input_size, hidden_size=hidden_size)
    predictor_optimizer = optim.SGD(predictor.parameters(), lr=0.00001, momentum=0.9, weight_decay=1.5e-4)
    predictor_criterion = nn.MSELoss()

    trained_model = train_predictor(predictor, encoder, dataloader, predictor_criterion, predictor_optimizer, device)

    # Optionally, save the final model
    save_model(trained_model, "predictor_final")
