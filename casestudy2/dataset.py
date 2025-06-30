import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import torch
import onnx
# from finn.util.test import get_test_model_trained
from brevitas.export import export_qonnx
# from qonnx.util.cleanup import cleanup as qonnx_cleanup
# from qonnx.core.modelwrapper import ModelWrapper
# from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
# from qonnx.transformation.infer_shapes import InferShapes
# from qonnx.transformation.fold_constants import FoldConstants
# from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
# from finn.util.basic import make_build_dir
# from finn.util.visualization import showInNetron
import os


data_dir = "./data"

# Create directories if they do not exist

os.makedirs(data_dir, exist_ok=True)
print(f"Data directory: {data_dir}")



def get_dataloaders(dataset_name='MNIST'):
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Currently only MNIST is supported.")