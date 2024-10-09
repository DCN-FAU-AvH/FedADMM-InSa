import torch
from torch import nn
import torchvision.models as models
from torchinfo import summary
from utils.resnet import *


def init_model(cfg, requires_grad=True):
    """Initialize a model."""
    if cfg.dataset == "linreg":
        model = LinRegModel(cfg)
    elif cfg.model == "cnn":
        model = CNN(cfg)
    elif cfg.model  == "resnet":
        model = resnet20()  # resnet20 for cifar10
        replace_bn_with_ln(model)
        model.loss = get_loss_func(cfg)
    else:
        raise ValueError(f"Invalid model.")
    cfg.device = torch.device(cfg.device)
    model.requires_grad_(requires_grad)  # requires gradients or not
    return model.to(cfg.device)


class LinRegModel(nn.Module):
    """Linear regression model."""

    def __init__(self, cfg):
        super().__init__()
        self.c_i = cfg.c_i
        self.linear = nn.Linear(cfg.linreg_dim_data, 1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x

    def loss(self, pred, labels):
        """Loss with l2 regularization."""
        loss_fn = nn.MSELoss()
        loss = 0.5 * loss_fn(pred, labels)
        l2_reg = 0
        for param in self.parameters():
            l2_reg += param.square().sum()
        loss += 0.5 * self.c_i * l2_reg
        return loss


class CNN(nn.Module):
    """CNN used in the FedAvg paper."""

    def __init__(self, cfg):
        super().__init__()
        dim_in_channels = {"mnist": 1, "cifar10": 3, "cifar100": 3}
        dim_in_fc = {"mnist": 1024, "cifar10": 1600, "cifar100": 1600}
        self.loss = get_loss_func(cfg)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in_channels[cfg.dataset], 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_in_fc[cfg.dataset], 512),
            nn.ReLU(True),
            nn.Linear(512, cfg.num_classes),  # output layer
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten the data from dim=1
        x = self.fc(x)
        return x


def get_activation(cfg):
    """Selects the activation function based on the cfg."""
    if cfg.activation == "relu":
        return nn.ReLU(inplace=True)  # to save memory
    elif cfg.activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Invalid activation function.")


def get_loss_func(cfg):
    """Selects the loss function based on the cfg."""
    if cfg.loss == "mse":
        return nn.MSELoss()
    elif cfg.loss == "cn":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")


def replace_bn_with_ln(module):
    """
    Replace bn in resnet with ln, see
    https://arxiv.org/abs/2308.09565
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=1, num_channels=num_features))
        else:
            replace_bn_with_ln(child)
