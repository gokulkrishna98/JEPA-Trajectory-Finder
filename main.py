from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import Encoder, Predictor, SimpleCNN, JEPAModel
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    """Load the trained Encoder and Predictor models."""
    device = get_device()

    # Initialize the encoder with the same architecture used in training
    encoder_cnn = SimpleCNN(embed_size=512, input_channel=2)  # Adjust input_channel as per your data
    encoder = Encoder(encoder_cnn).to(device)

    # Load the encoder weights
    encoder_path = "encoder.pth"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()  # Set encoder to evaluation mode

    # Initialize the predictor with the same architecture used in training
    predictor = Predictor(input_size=2, hidden_size=1024).to(device)

    # Load the predictor weights
    predictor_path = "predictor.pth"
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor.eval()  # Set predictor to evaluation mode

    model = JEPAModel(encoder, predictor)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
