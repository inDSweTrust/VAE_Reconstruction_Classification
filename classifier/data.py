import torch
from torch.utils.data import DataLoader, random_split

from config import CFG


class CIFAR10_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        # Load CIFAR10 latent vectors

        self.data = dataset[:, :-1].to(device)
        self.targets = dataset[:, -1].to(torch.int32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        return img, target


def prepare_dataloaders(latent_dir):
    cifar10 = torch.load(latent_dir)
    cifar10_train, cifar10_val = random_split(cifar10, [40000, 10000])

    train_data = CIFAR10_Dataset(cifar10_train.dataset, CFG["device"])
    val_data = CIFAR10_Dataset(cifar10_val.dataset, CFG["device"])

    trainloader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=CFG["batch_size"], shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=CFG["batch_size"], shuffle=False
    )
    return trainloader, valloader
