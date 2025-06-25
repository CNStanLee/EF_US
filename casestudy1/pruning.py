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
import copy
import torch.nn.utils.prune as prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def analyze_model_sparsity(model):
    param_details = []
    

    for name, module in model.named_modules():
        if not name: 
            continue
            
        layer_type = module.__class__.__name__
        
        for param_name, param in module.named_parameters(recurse=False):
            param_count = param.numel()
            non_zero_count = torch.count_nonzero(param).item()
            sparsity = 100 * (1 - non_zero_count / param_count) if param_count > 0 else 0
            param_details.append((f"{name}.{param_name}", layer_type, param_count, non_zero_count, sparsity))
    
    if param_details:
        max_path = max(len(str(d[0])) for d in param_details)
        max_type = max(len(str(d[1])) for d in param_details)
        max_count = max(len(f"{d[2]:,}") for d in param_details)  # 带千位分隔符
        max_nonzero = max(len(f"{d[3]:,}") for d in param_details)
        
        print(f"{'Parameter Path':<{max_path}} | {'Layer Type':<{max_type}} | {'Param Count':>{max_count}} | {'Non-zero':>{max_nonzero}} | {'Sparsity (%)':>10}")
        print("-" * (max_path + max_type + max_count + max_nonzero + 30))  # 动态分隔线长度
        
        for detail in param_details:
            print(f"{detail[0]:<{max_path}} | {detail[1]:<{max_type}} | {detail[2]:>{max_count},} | {detail[3]:>{max_nonzero},} | {detail[4]:>10.2f}%")
    
    if param_details:
        total_params = sum(d[2] for d in param_details)
        total_non_zero = sum(d[3] for d in param_details)
        total_sparsity = 100 * (1 - total_non_zero / total_params) if total_params > 0 else 0
        
        print("-" * (max_path + max_type + max_count + max_nonzero + 30))
        print(f"{'TOTAL':<{max_path}} | {'-':<{max_type}} | {total_params:>{max_count},} | {total_non_zero:>{max_nonzero},} | {total_sparsity:>10.2f}%")
        compression_ratio = total_params / total_non_zero if total_non_zero > 0 else float('inf')
        print(f"\nCompression Ratio (pruning only): {compression_ratio:.2f}x")
        
        effective_compression = 4 * compression_ratio  # 4 = 32/8
        print(f"Effective Compression (pruning + quantization): {effective_compression:.2f}x")


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
    """
    改进版：确保能正确剪枝所有层类型
    
    参数:
        verbose: 打印剪枝详细信息
    """
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