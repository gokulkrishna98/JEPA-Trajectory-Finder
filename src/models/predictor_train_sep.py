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
from encoder import Encoder, SimpleCNN

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def save_model(model, epoch, save_path="checkpoints_pred_expand", file_name="pred"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


# todo: have to remove the use of `use expander`, it for testing
def train_predictor(pred, enc, dataloader, criterion, optimizer, device, 
                    use_expander=False, epochs=10):
    # keeping encoder in eval mode
    pred, enc = pred.to(device), enc.to(device)

    # freezing the encoder and setting it to evaluation mode
    enc.eval()
    for param in enc.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Processing batch"):
            ## shape of [ s = (B, L+1, C, H, W)]  [a = (B, L, 2)]
            s, a = batch
            s, a = s.to(device), a.to(device)

            ## initial observation
            o = s[:, 0, :, :, :]
            # o = torch.cat([o, o[:, 1:2, :, :]], dim=1)
            with torch.no_grad():
                x, z = enc(o)
                so = z if use_expander else x
            
            ## initializing predictor h,c
            ## check randn instead of zeros
            co = torch.zeros(so.shape).to(device)
            pred.set_hc(so, co)
            
            ## forward inference for training.
            loss ,L = 0, a.shape[1]
            for i in range(L):
                sy_hat = pred(a[:, i, :])
                si = s[:, i+1, :, :, :]
                # si = torch.cat([si, si[:, 1:2, :, :]], dim=1)
                with torch.no_grad():
                    x, z = enc(si)
                    sy = z if use_expander else x
                loss += criterion(sy_hat, sy)
            
            ## back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## clearing h,c in lstm 
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
    encoder = SimpleCNN(512, 2) 
    encoder = Encoder(encoder).to(device)

    input_size = 2
    hidden_size = 1024
    encoder.load_state_dict(torch.load("encoder__encoder_cnn_final.pth"))

    predictor = Predictor(input_size=input_size, hidden_size=hidden_size)
    predictor_optimizer = optim.SGD(predictor.parameters(), lr=0.001, momentum=0.9, weight_decay=1.5e-4)
    predictor_criterion = nn.MSELoss()

    trained_model = train_predictor(predictor, encoder, dataloader, predictor_criterion,
                predictor_optimizer, device, use_expander=True)
    # Optionally, save the final model
    save_model(trained_model, "predictor_final")
