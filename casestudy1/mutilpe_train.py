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
from finn.util.test import get_test_model_trained
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import os

from models import get_model
from dataset import get_dataloaders
from train import train_try

def main():
    #train_try(model_name='2c3f', w=8, a=8, epochs=10, random_seed=1998)

    w_set = [1, 2, 4, 8]
    a_set = [1, 2, 4, 8]
    # model_names = ['2c3f', 'tfc', 'sfc', 'lfc']
    model_names = ['tfc', 'sfc', 'lfc']
    epochs = 500
    for model_name in model_names:
        print(f"Training model {model_name} with different weight and activation bit widths")
        # Iterate through all combinations of weight and activation bit widths
        for w in w_set:
            for a in a_set:
                print(f"Training model with weight bit width {w} and activation bit width {a}")
                train_try(model_name = model_name, w=w, a=a, epochs=epochs, random_seed=1998)
                print(f"Finished training model with weight bit width {w} and activation bit width {a}")


if __name__ == "__main__":
    main()