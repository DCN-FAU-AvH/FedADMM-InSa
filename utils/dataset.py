import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset
from scipy.stats import t as t_distri

import os, shutil
import urllib.request
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def init_dataset(cfg):
    """Load and split datasets based on the parameters in cfg."""
    if torch.cuda.is_available():  # GPU training
        # path to download the dataset
        path = "../../dataset"
    else:  # cpu training
        path = r"D:\FAUBox\code\dataset"

    if cfg.dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans)
        cfg.num_classes = 10
    elif cfg.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        trans_tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trans_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.CIFAR10(root=path, train=True, download=True, transform=trans_tr)
        dataset_test = datasets.CIFAR10(root=path, train=False, download=True, transform=trans_te)
        cfg.num_classes = 10
    elif cfg.dataset == "linreg":
        dataset_train = LinReg(num_data=50000, dim_data=5000)
        dataset_test = LinReg(num_data=5, dim_data=5000)
    else:
        opts = ["mnist", "linreg", "cifar10"]
        raise ValueError(f"Invalid dataset. Choose from {opts}")

    # data separation
    if cfg.dataset == "linreg" or cfg.iid:
        split_dict = split_iid(dataset_train, cfg.m)
    else:
        split_dict = split_noniid(dataset_train, cfg)
    dataset = {"train": dataset_train, "test": dataset_test, "split_dict": split_dict}
    cfg.alpha = cal_alpha(cfg, dataset)  # percentage of clients' datasets
    return dataset


class LinReg(Dataset):
    """Generate a dataset for linear regression tests."""

    def __init__(self, num_data=100, dim_data=100, frac_1=1 / 3, frac_2=1 / 3):
        self.data, self.labels = self._get_data(num_data, dim_data, frac_1, frac_2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def _get_data(self, num_data, dim_data, frac_1, frac_2):
        """Generate data in three different distributions."""
        # standard normal distribution
        A = np.random.randn(num_data, dim_data)
        b = np.random.randn(num_data, 1)
        if frac_1 + frac_2 > 0:
            # student t distribution
            num_1 = np.ceil(frac_1 * num_data).astype(int)
            A[:num_1, :] = t_distri.rvs(5, size=(num_1, dim_data))
            b[:num_1] = t_distri.rvs(5, size=(num_1, 1))
            # uniform distribution
            num_2 = np.ceil(frac_2 * num_data).astype(int)
            A[num_1 : num_1 + num_2, :] = np.random.uniform(-5, 5, (num_2, dim_data))
            b[num_1 : num_1 + num_2] = np.random.uniform(-5, 5, (num_2, 1))
            # randomize data
            I = np.random.permutation(num_data)
            A = A[I, :]
            b = b[I, :]
        return torch.tensor(A, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def split_iid(dataset, num_clients):
    """Split the dataset in an iid way."""
    data_per_client = len(dataset) // num_clients  # The same amount of data per customer.
    data_indices = range(len(dataset))
    # Create a split dict, where key: client idx, value: client's data indices.
    split_dict = {i: np.array([], dtype="int32") for i in range(num_clients)}
    for i in range(num_clients):
        # allocate data
        split_dict[i] = np.random.choice(data_indices, data_per_client, replace=False)
        data_indices = list(set(data_indices) - set(split_dict[i]))  # remaining data
    return split_dict


def split_noniid(dataset, cfg):
    """
    Split the dataset in a non-iid way,
    where each client only has a limited categoris of data.
    """
    if cfg.dataset == "cifar10":
        shards_per_client = 5
    else:
        shards_per_client = 2
    num_clients = cfg.m
    data_per_client = len(dataset) // num_clients  # The same amount of data per customer.
    data_per_shard = data_per_client // shards_per_client
    num_shards = num_clients * shards_per_client
    shards = [i for i in range(num_shards)]
    # Create a split dict, where key: client idx, value: client's data indices.
    split_dict = {i: np.array([], dtype="int32") for i in range(num_clients)}
    # Load and sort labels.
    labels = np.asarray([label for _, label in dataset]).argsort()
    # Load indices of data in the order of sorted labels.
    data_sorted = np.arange(len(dataset))[labels]
    # Split the dataset to clients.
    for i in range(num_clients):
        # Randomly select shards for client i.
        shards_i = set(np.random.choice(shards, shards_per_client, replace=False))
        # Allocate data in selected shards to client i.
        for shard_i in shards_i:
            start = shard_i * data_per_shard  # index of the first data in shard_i
            end = start + data_per_shard  # index of the last data in shard_i
            split_dict[i] = np.concatenate((split_dict[i], data_sorted[start:end]))
        shards = list(set(shards) - shards_i)  # Get indices of remaining shards.
    return split_dict


class DatasetSplit(Dataset):
    """Return client i's dataset."""

    def __init__(self, dataset, client_i):
        self.dataset = dataset["train"]  # whole training set
        self.indices_i = list(dataset["split_dict"][client_i])  # list of indices of client's data

    def __len__(self):
        return len(self.indices_i)

    def __getitem__(self, index):
        data_index = self.indices_i[index]
        data, label = self.dataset[data_index]
        return data, label


def cal_alpha(cfg, dataset):
    "Calculate the percentage of each client's dataset."
    alpha = []
    for i in range(cfg.m):
        dataset_i = DatasetSplit(dataset, i)
        alpha_i = len(dataset_i) / len(dataset["train"])  # percentage of the client's data
        alpha.append(alpha_i)
    return alpha


def get_subset(dataset, set_size, classes):
    """Get a subset with the same number of training data in each class."""
    num_per_class = set_size // classes
    class_indices = {i: [] for i in range(classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < num_per_class:
            class_indices[label].append(idx)
        if all(len(indices) == num_per_class for indices in class_indices.values()):
            break
    subset_indices = sum(class_indices.values(), [])
    return Subset(dataset, subset_indices)
