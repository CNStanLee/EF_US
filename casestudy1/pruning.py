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
from dataclasses import dataclass
import copy
import torch.nn.utils.prune as prune
from models import get_model
from collections import namedtuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# 定义一个简单的命名元组来存储参数信息
ParameterInfo = namedtuple('ParameterInfo', [
    'path',        # 完整参数路径 (如 "conv1.weight")
    'layer_type',  # 层类型 (如 "Conv2d")
    'param_count', # 参数总数
    'non_zero',    # 非零参数数量
    'sparsity'     # 稀疏度百分比
])

def analyze_model_sparsity(model):
    """Analyze sparsity of a PyTorch model
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        List of ParameterInfo namedtuples containing sparsity information
    """
    param_details = []
    
    for name, module in model.named_modules():
        if not name:  # Skip the root module which has empty name
            continue
            
        layer_type = module.__class__.__name__
        
        for param_name, param in module.named_parameters(recurse=False):
            param_count = param.numel()
            non_zero_count = torch.count_nonzero(param).item()
            sparsity = 100 * (1 - non_zero_count / param_count) if param_count > 0 else 0
            
            full_path = f"{name}.{param_name}"
            param_details.append(
                ParameterInfo(
                    path=full_path,
                    layer_type=layer_type,
                    param_count=param_count,
                    non_zero=non_zero_count,
                    sparsity=sparsity
                )
            )
    
    # Print the results in a formatted table
    if param_details:
        max_path = max(len(info.path) for info in param_details)
        max_type = max(len(info.layer_type) for info in param_details)
        max_count = max(len(f"{info.param_count:,}") for info in param_details) 
        max_nonzero = max(len(f"{info.non_zero:,}") for info in param_details)
        
        # Header
        print(f"{'Parameter Path':<{max_path}} | {'Layer Type':<{max_type}} | {'Param Count':>{max_count}} | {'Non-zero':>{max_nonzero}} | {'Sparsity (%)':>10}")
        print("-" * (max_path + max_type + max_count + max_nonzero + 30))
        
        # Rows
        for info in param_details:
            print(f"{info.path:<{max_path}} | {info.layer_type:<{max_type}} | {info.param_count:>{max_count},} | {info.non_zero:>{max_nonzero},} | {info.sparsity:>10.2f}%")
    
    # Print summary
    if param_details:
        total_params = sum(info.param_count for info in param_details)
        total_non_zero = sum(info.non_zero for info in param_details)
        total_sparsity = 100 * (1 - total_non_zero / total_params) if total_params > 0 else 0
        
        print("-" * (max_path + max_type + max_count + max_nonzero + 30))
        print(f"{'TOTAL':<{max_path}} | {'-':<{max_type}} | {total_params:>{max_count},} | {total_non_zero:>{max_nonzero},} | {total_sparsity:>10.2f}%")
        
        compression_ratio = total_params / total_non_zero if total_non_zero > 0 else float('inf')
        print(f"\nCompression Ratio (pruning only): {compression_ratio:.2f}x")
        
        #effective_compression = 4 * compression_ratio  # 4 = 32/8 (assuming 32-bit to 8-bit quantization)
        #print(f"Effective Compression (pruning + quantization): {effective_compression:.2f}x")

    return param_details




def apply_l1_pruning_type(model, layer_type, pruning_percentage=0.5):
    # layer_type: QuantLinear or QuantConv2d
    model_tmp = copy.deepcopy(model).to(device)
    for name, module in model_tmp.named_modules():
        if isinstance(module, layer_type):
            # if the para number <100, skip pruning
            if module.weight.numel() > 10:
                prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
                prune.remove(module, 'weight')  
    return model_tmp



def apply_l1_pruning_by_param_path(model, param_path, pruning_percentage=0.5, verbose=False):
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    
    parts = param_path.split('.')
    param_name = parts[-1]
    module_path = '.'.join(parts[:-1])
    
    target_module = model_copy
    if module_path:  
        for part in module_path.split('.'):
            try:
                target_module = getattr(target_module, part)
            except AttributeError:
                raise ValueError(f"Module path '{module_path}' not found (failed at '{part}')")
    
    if not hasattr(target_module, param_name):
        raise ValueError(f"Parameter '{param_name}' not found in module '{module_path}'")
    
    param = getattr(target_module, param_name)
    if verbose:
        print(f"Pruning {param_path} | Shape: {tuple(param.shape)} | Elements: {param.numel()}")

    prune.l1_unstructured(target_module, name=param_name, amount=pruning_percentage)
    prune.remove(target_module, param_name)
    
    return model_copy

# pruning method:
# l1
# l2
# random
# magnitude
# SNIP

def main():
    # input parameters
    w = 1
    a = 2
    model_name = "2c3f"
    model_weight = "./model/best_2c3f_w1_a2_10.pth"

    # Analyze model sparsity
    print("Analyzing model sparsity...")
    model = get_model(model_name, w, a)
    model.load_state_dict(torch.load(model_weight))
    model.to(device)

    sparsity_info = analyze_model_sparsity(model)

    # 访问第一个参数的信息
    first_param = sparsity_info[0]
    print(f"Path: {first_param.path}")
    print(f"Type: {first_param.layer_type}")
    print(f"Sparsity: {first_param.sparsity}%")

    # 'path',        # 完整参数路径 (如 "conv1.weight")
    # 'layer_type',  # 层类型 (如 "Conv2d")
    # 'param_count', # 参数总数
    # 'non_zero',    # 非零参数数量
    # 'sparsity'     # 稀疏度百分比

    # 遍历所有参数

    # deep copy the model
    pmodel = copy.deepcopy(model).to(device)

    for param in sparsity_info:
        if param.param_count > 150: 
            pmodel = apply_l1_pruning_by_param_path(pmodel, param.path, pruning_percentage=0.5, verbose=True)

    sparsity_info = analyze_model_sparsity(pmodel)

    # Stage 1: sensitivity analysis
    print("\nStage 1: Sensitivity Analysis")
    # prune each layer seprately from 10% to 90% with step 10% and check accuracy decrease

    # Stage 2: With a tolerance, prune each layer with their specific pruning rate
    print("\nStage 2: Pruning with Specific Rates")

    # Stage 3: Re-train the model, try to fit accuracy dropdown acceptable

    # Stage 4: if accuracy is not acceptable, try to change tolerance and pruning rate, do 4 again

    # Stage 5: export the model to ONNX and FINN format




if __name__ == "__main__":
    main()