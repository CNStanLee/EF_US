import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import onnx
import numpy as np
import urllib.request

data_dir = "./data"

# Create directories if they do not exist

os.makedirs(data_dir, exist_ok=True)
print(f"Data directory: {data_dir}")



def get_dataloaders(dataset_name='MNIST', batch_size=256):
    if dataset_name == 'MNIST':
        # Load the MNIST dataset
        transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]
        )
        full_train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        train_size = int(0.8 * len(full_train_set))
        val_size = len(full_train_set) - train_size

        train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

        print("Dataset split complete:")
        print(f"Training set: {len(train_set)} samples")
        print(f"Validation set: {len(val_set)} samples")
        print(f"Test set: {len(test_set)} samples")
        return train_loader, val_loader, test_loader
    elif dataset_name == 'UNSW':
        def get_preqnt_dataset(data_dir: str, train: bool):
            # wget -O data/unsw_nb15_binarized.npz https://zenodo.org/record/4519767/files/unsw_nb15_binarized.npz?download=1
            # Download the dataset if it does not exist
            if not os.path.exists(data_dir + "/unsw_nb15_binarized.npz"):
                url = "https://zenodo.org/record/4519767/files/unsw_nb15_binarized.npz?download=1"
                urllib.request.urlretrieve(url, data_dir + "/unsw_nb15_binarized.npz")
            unsw_nb15_data = np.load(data_dir + "/unsw_nb15_binarized.npz")
            if train:
                partition = "train"
            else:
                partition = "test"
            part_data = unsw_nb15_data[partition].astype(np.float32)
            part_data = torch.from_numpy(part_data)
            part_data_in = part_data[:, :-1]
            part_data_out = part_data[:, -1]
            return TensorDataset(part_data_in, part_data_out)

        train_quantized_dataset = get_preqnt_dataset("./data", True)
        test_quantized_dataset = get_preqnt_dataset("./data", False)
        # split the training dataset into training and validation sets
        train_size = int(0.8 * len(train_quantized_dataset))
        val_size = len(train_quantized_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(train_quantized_dataset, [train_size, val_size])   
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_quantized_dataset, batch_size=256, shuffle=False)
        print("Dataset split complete:")
        print(f"Training set: {len(train_set)} samples")
        print(f"Validation set: {len(val_set)} samples")
        print(f"Test set: {len(test_quantized_dataset)} samples")
        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Currently only MNIST and UNSW are supported.")