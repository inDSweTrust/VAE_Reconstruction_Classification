import torch

import torchvision

from torchvision import transforms
from torchvision.datasets import CIFAR10

from config import CFG


class CIFAR10_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        # Load CIFAR10 as numpy array
        if train:
            self.cifar10 = CIFAR10(
                root="./data", train=True, download=True, transform=None
            )
        else:
            self.cifar10 = CIFAR10(
                root="./data", train=False, download=True, transform=None
            )

        self.transform = transform

    def __len__(self):
        return len(self.cifar10.targets)

    def __getitem__(self, idx):
        img = self.cifar10.data[idx]

        target = self.cifar10.targets[idx]

        if self.transform:
            img = self.transform(img)
            img = img.view(3, 32, 32)

        return img, target


def prepare_dataloaders():
    transform = transforms.ToTensor()

    cifar10_train = CIFAR10_Dataset(True, transform)
    cifar10_test = CIFAR10_Dataset(False, transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=cifar10_train, batch_size=CFG["batch_size"], shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=cifar10_test, batch_size=CFG["batch_size"], shuffle=False
    )
    return train_loader, test_loader
