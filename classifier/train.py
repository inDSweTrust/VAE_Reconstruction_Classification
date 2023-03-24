import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from tqdm import tqdm

from data import prepare_dataloaders
from nn import LinearClassifier
from config import CFG


class TrainClassifier:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        trainloader,
        testloader,
        epochs,
        device,
        writer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.device = device
        self.writer = writer

        self.running_loss_train = 0.0
        self.running_loss_test = 0.0

    def train_step(self, data, target):
        data = data.to(self.device)
        target = target.type(torch.LongTensor)
        target = target.to(self.device)

        self.optimizer.zero_grad()

        out = self.model(data)
        loss = self.criterion(out, target)
        self.running_loss_train += loss.item()

        loss.backward()
        self.optimizer.step()

    def eval_step(self, data, target):
        data = data.to(self.device)
        target = target.type(torch.LongTensor)
        target = target.to(self.device)

        out = self.model(data)
        loss = self.criterion(out, target)
        self.running_loss_test += loss.item()

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")

            i = 0
            self.running_loss_train = 0.0

            for data, target in tqdm(self.trainloader):
                self.train_step(data, target)

                i += 1

            train_loss = self.running_loss_train / i
            self.writer.add_scalar("Loss/train", train_loss, epoch)

            print("TRAIN LOSS", train_loss)

            self.evaluate(self.testloader, epoch)
            self.model.train()

    def evaluate(self, dataloader, epoch):
        self.model.eval()
        self.running_loss_test = 0.0
        i = 0

        with torch.no_grad():
            for data, target in tqdm(dataloader):
                self.eval_step(data, target)
                i += 1

            eval_loss = self.running_loss_test / i
            self.writer.add_scalar("Loss/test", eval_loss, epoch)

            print("TEST LOSS", eval_loss)

            eval_acc = self.accuracy(dataloader.dataset)
            self.writer.add_scalar("Acc/test", eval_acc, epoch)

            print("TEST ACC", eval_acc)

    def accuracy(self, dataset):
        data = dataset.data
        targets = dataset.targets

        with torch.no_grad():
            logits = self.model(data)  # forwards pass to compute log probabilities

        preds = torch.argmax(logits, dim=1)
        num_correct = torch.sum(targets == preds)
        acc = (num_correct * 1.0) / len(dataset)

        return acc.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="path to encoded latent space .pt file")
    parser.add_argument("-d", "--latent_dim", help="latent space dimension")

    args = parser.parse_args()

    latent_dir = args.path
    latent_dim = int(args.latent_dim)

    model = LinearClassifier(input_dim=latent_dim).to(CFG["device"])
    summary(model, (latent_dim,))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    trainloader, valloader = prepare_dataloaders(latent_dir)

    writer = SummaryWriter("./runs/cifar10_v1")

    classifier = TrainClassifier(
        model,
        optimizer,
        criterion,
        trainloader,
        valloader,
        CFG["epochs"],
        CFG["device"],
        writer,
    )
    classifier.train()

    writer.flush()
