import numpy as np
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from dataset import TrajectoryDataset
from encoder import Encoder

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_byol_transforms(mean, std):
    # Define the first augmentation pipeline
    transformT = tr.Compose([
        tr.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        tr.RandomRotation(degrees=90),  # Random rotation
        tr.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),  # Gaussian blur
        tr.Normalize(mean, std),  # Normalize for 2 channels
    ])

    # Define a slightly different second augmentation pipeline
    transformT1 = tr.Compose([
        tr.RandomVerticalFlip(p=0.5),  # Random vertical flip
        tr.RandomRotation(degrees=45),  # Different random rotation
        tr.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 1.5)),  # Gaussian blur with smaller kernel
        tr.Normalize(mean, std),  # Normalize for 2 channels
    ])
    
    return transformT, transformT1


def off_diagonal(matrix):
    """
    Extracts the off-diagonal elements of a square matrix.
    
    Args:
        matrix (torch.Tensor): A square matrix of shape (D, D).
    
    Returns:
        torch.Tensor: A tensor containing all off-diagonal elements.
    """
    # Create a mask for off-diagonal elements
    n = matrix.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=bool, device=matrix.device)
    
    # Use the mask to extract off-diagonal elements
    off_diag_elements = matrix[off_diag_mask]
    return off_diag_elements


def criterion(x, y, invar = 25, mu = 25, nu = 1, epsilon = 1e-4):
    bs = x.size(0)
    emb = x.size(1)

    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    invar_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_z_a = (x.T @ x) / (bs - 1)
    cov_z_b = (y.T @ y) / (bs - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / emb + off_diagonal(cov_z_b).pow_(2).sum() / emb

    # print(f"invar_loss: {invar_loss.item()}")
    # print(f"var_loss: {var_loss.item()}")
    # print(f"cov_loss: {cov_loss.item()}")

    loss = invar*invar_loss + mu*var_loss + nu*cov_loss
    return loss

def compute_mean_and_std(dataloader, is_channelsize3 = True):
    num_channels = 2  # Assuming you have 2 channels
    pixel_sum = [0] * num_channels
    pixel_squared_sum = [0] * num_channels
    total_pixels = 0

    # Iterate through the dataset
    for state, _ in dataloader:
        # Iterate through each channel
        for channel in range(num_channels):
            channel_data = state[:, :, channel, :, :].reshape(-1)  # Flatten the current channel
            pixel_sum[channel] += channel_data.sum().item()
            pixel_squared_sum[channel] += (channel_data ** 2).sum().item()
        
        # Total number of pixels per channel (all images combined)
        total_pixels += state.size(0) * state.size(1) * state.size(3) * state.size(4)

    # Calculate mean and std for each channel
    mean = [pixel_sum[c] / total_pixels for c in range(num_channels)]
    std = [
        np.sqrt((pixel_squared_sum[c] / total_pixels) - (mean[c] ** 2))
        for c in range(num_channels)
    ]

    # print(f"Mean per channel: {mean}")
    # print(f"Std per channel: {std}")
    
    # Adding a 3rd dimension
    if is_channelsize3:
        mean.append(mean[1])
        std.append(std[1])

    return mean, std

def save_model(model, epoch, save_path="checkpoints", file_name="encoder_"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


def train_model(dataloader, model, epochs, device, transformation1, transformation2, step = 1):
    for epoch in range(epochs):
        total_loss = 0
        #ind = 0
        for batch in dataloader:
            state, _ = batch
            state = state.to(device)
            for i in range(state.size(1)):
                img = state[:, i, :, :, :]
                img = torch.cat([img, img[:, 1:2, :, :]], dim=1)

                x0 = transformation1(img)
                x1 = transformation2(img)

                x0 = x0.to(device)
                x1 = x1.to(device)

                z0 = model(x0)
                z1 = model(x1)

                loss = criterion(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print(f"batch: {ind}")
                avg_loss = total_loss / len(dataloader)
                # ind = ind + 1
        # Save model checkpoint
        if epoch % step == 0:
            save_model(model, epoch)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    print("Training completed.")
    return model

def get_encoder_loss(model, img, transformation1, transformation2, criterion):
    x0 = transformation1(img)
    x1 = transformation2(img)
    _, z0 = model(state=x0)
    _, z1 = model(state=x1)

    loss = criterion(z0, z1)
    return loss

if __name__ == "__main__":
    print("Training main function")
    device = get_device()

    dataset = TrajectoryDataset(
        data_dir = "../dataset",
        states_filename = "states1.npy",
        actions_filename = "actions1.npy",
        s_transform = None,
        a_transform = None,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    
    model = Encoder(backbone)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum= 0.9, weight_decay=1.5e-4)

    mean, std = compute_mean_and_std(dataloader)

    transformation1, transformation2  = get_byol_transforms(mean, std)

    trained_model = train_model(dataloader, model, 1, device, transformation1, transformation2)

    # Optionally, save the final model
    save_model(trained_model, "encoder_final")
