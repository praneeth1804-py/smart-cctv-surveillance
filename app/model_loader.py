import torch
from model_training import ConvAutoencoder
import os

device = torch.device("cpu")

model = ConvAutoencoder()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "anomaly_model.pth")

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()