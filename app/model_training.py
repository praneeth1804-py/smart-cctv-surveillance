import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import glob


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(15,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,15,2,stride=2),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training on:", device)

    model = ConvAutoencoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    DATASET_DIR = "../cached_dataset"
    dataset_files = sorted(
        glob.glob(os.path.join(DATASET_DIR,"*.pt"))
    )

    epochs = 30
    os.makedirs("../models",exist_ok=True)

    for epoch in range(epochs):

        total_loss = 0

        for part in dataset_files:

            print(f"\nLoading {part}")

            data = torch.load(part)

            loader = DataLoader(
                data,
                batch_size=32,
                shuffle=True
            )

            for batch_idx,images in enumerate(loader):

                images = images.to(device)

                output = model(images)

                loss = criterion(output,images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch+1} | "
                        f"{os.path.basename(part)} | "
                        f"Batch {batch_idx} | "
                        f"Loss {loss.item():.6f}"
                    )

            del data

        print(
            f"\n✅ Epoch {epoch+1} Complete "
            f"| Loss {total_loss:.4f}"
        )

        torch.save(
            model.state_dict(),
            f"../models/anomaly_epoch_{epoch+1}.pth"
        )

    torch.save(
        model.state_dict(),
        "../models/anomaly_model.pth"
    )

    print("\n✅ TRAINING FINISHED")


if __name__=="__main__":
    train()