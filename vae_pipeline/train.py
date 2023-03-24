import os
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure
from torchsummary import summary

from tqdm import tqdm

from config import CFG
from data import prepare_dataloaders
from vae import ConvVAE


class TrainVAE(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        device,
        trainloader,
        testloader,
        epochs,
        latent_dim,
        gen=True,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.gen = gen
        self.latent_dim = latent_dim

        self.latent_data = []
        self.latent_targets = []

        self.sample = torch.randn(CFG["batch_size"], latent_dim).to(device)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            cumul_loss = 0.0

            for img, target in tqdm(self.trainloader):
                img = img.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                out, z, mu, logvar = self.model(img)
                loss = self.recon_loss(out, img, mu, logvar)

                if epoch == self.epochs - 1:
                    self.latent_data.append(z)
                    self.latent_targets.append(target)

                cumul_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            train_epoch_loss = cumul_loss / len(self.trainloader.dataset)
            print(f"Train Loss: {train_epoch_loss:.4f}")

            val_epoch_loss, val_epoch_acc = self.validate(epoch)
            print(f"Val Loss: {val_epoch_loss:.4f}")
            print(f"Val Acc: {val_epoch_acc:.4f}")

        self.save_output()

    def validate(self, epoch):
        self.model.eval()
        cumul_loss = 0.0
        cumul_acc = 0.0

        isExist = os.path.exists(os.path.join(root_dir, "recon"))
        if not isExist:
            os.mkdir(os.path.join(root_dir, "recon"))

        if self.gen:
            isExist = os.path.exists(os.path.join(root_dir, "gen"))
            if not isExist:
                os.mkdir(os.path.join(root_dir, "gen"))

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(self.testloader),
                total=int(len(self.testloader.dataset) / self.testloader.batch_size),
            ):
                data, _ = data
                data = data.to(self.device)

                recon, z, mu, logvar = self.model(data)
                loss = self.recon_loss(recon, data, mu, logvar)
                acc = self.recon_accuracy(recon, data)
                cumul_loss += loss.item()
                cumul_acc += acc.item()

                # Image generation
                if self.gen:
                    gen = self.model.decode(self.sample)
                    resultsample = gen * 0.5 + 0.5
                    resultsample = resultsample.cpu()

                # save the last batch input and output of every epoch
                if (
                    i
                    == int(len(self.testloader.dataset) / self.testloader.batch_size)
                    - 1
                ):
                    num_rows = 8

                    both = torch.cat(
                        (
                            data.view(CFG["batch_size"], 3, 32, 32)[:8],
                            recon.view(CFG["batch_size"], 3, 32, 32)[:8],
                        )
                    )

                    save_image(
                        both.cpu(),
                        os.path.join(root_dir, f"recon/recon{epoch}.png"),
                        nrow=num_rows,
                    )

                    if self.gen:
                        img_gen = gen.view(CFG["batch_size"], 3, 32, 32)
                        save_image(
                            img_gen.cpu(),
                            os.path.join(root_dir, f"gen/gen{epoch}.png"),
                            nrow=num_rows,
                        )

        val_loss = cumul_loss / len(self.testloader.dataset)
        val_acc = cumul_acc / len(self.testloader.dataset)

        return val_loss, val_acc

    def recon_loss(self, recon_x, x, mu, logvar):
        """
        Reconstruction loss: BCE and KL-Divergence.

        """
        criterion = nn.BCELoss(reduction="sum")
        BCE = criterion(recon_x, x) / x.size(0)

        KLD = ((mu**2 + logvar.exp() - 1 - logvar) / 2).mean()

        return BCE + KLD

    def recon_accuracy(self, recon_x, x):
        """
        Reconstruction capacity: SSIM

        """
        ssim = StructuralSimilarityIndexMeasure().to(self.device)

        return ssim(recon_x, x)

    def save_output(self):
        self.latent_data = torch.cat((self.latent_data), dim=0)
        self.latent_targets = torch.cat((self.latent_targets), dim=0).reshape(-1, 1)

        latent_out = torch.cat((self.latent_data, self.latent_targets), dim=1)

        torch.save(
            latent_out, os.path.join(root_dir, f"latent_data_dim{self.latent_dim}.pt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-rd", "--root", help="root directory")
    parser.add_argument("-dim", "--latent_dim", help="latent space dimension")

    args = parser.parse_args()

    root_dir = args.root
    latent_dim = int(args.latent_dim)

    model = ConvVAE(3, 128, latent_dim, 2, 1).to(CFG["device"])
    summary(model, (3, 32, 32))

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    train_loader, test_loader = prepare_dataloaders()

    trainer = TrainVAE(
        model,
        optimizer,
        CFG["device"],
        train_loader,
        test_loader,
        CFG["epochs"],
        latent_dim,
    )

    trainer.train()
