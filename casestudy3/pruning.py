import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from tqdm import tqdm
import os
import onnx
import copy
import numpy as np
from collections import namedtuple, defaultdict
from dataset import get_dataloaders
from models import get_model
from brevitas.nn import QuantConv2d, QuantLinear
import json
import csv
from collections import namedtuple
from trainer import Trainer

import warnings

ParameterInfo = namedtuple('ParameterInfo', [
    'path',       
    'layer_type',  
    'param_count', 
    'non_zero',    
    'sparsity'     
])
PruningDecision = namedtuple('PruningDecision', ['layer_path', 'pruning_rate'])
PruningResult = namedtuple('PruningResult', ['layer_path', 'layer_type', 'param_count','pruning_percentage', 'test_accuracy', 'accuray_drop'])

def freeze_zero_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            mask = (param != 0).float()
            param.register_hook(lambda grad, mask=mask: grad * mask)

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
        
        # if layer type name contain _, skip analysis
        if '_' in layer_type:
            continue

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
        
    return param_details

def global_magnitude_prune_with_min(model, target_sparsity):
    """
    全局magnitude剪枝，确保每层剪枝后参数不少于64个
    Args:
        model: 待剪枝的模型
        target_sparsity: 目标稀疏度(0-1之间)
    """
    # 步骤1: 收集所有符合条件的权重
    all_weights = []
    layer_info = {}  # 存储层信息: {module: (weight_tensor, min_retain)}
    
    for name, module in model.named_modules():
        # 只处理QuantLinear和QuantConv2d层
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
            
        # 跳过小参数层(<=64)
        if not hasattr(module, 'weight') or module.weight is None:
            continue
            
        weight = module.weight.data
        param_count = weight.numel()
        
        if param_count <= 64:
            # 小层不参与剪枝
            continue
        
        # 计算该层最多可剪枝的数量（确保保留至少64个参数）
        max_prune = param_count - 64
        layer_info[module] = (weight, max_prune)
        
        # 收集权重
        all_weights.append(weight.view(-1))
    
    if not all_weights:
        print("No parameters available for pruning.")
        return
    
    # 步骤2: 计算全局剪枝阈值
    all_weights = torch.cat(all_weights)
    total_params = len(all_weights)
    num_prune = int(target_sparsity * total_params)
    
    if num_prune == 0:
        print("Pruning amount is zero, skipping.")
        return
    
    # 找到全局阈值 (绝对值最小的num_prune个权重中的最大值)
    sorted_abs_weights = torch.sort(torch.abs(all_weights)).values
    global_threshold = sorted_abs_weights[num_prune - 1] if num_prune > 0 else 0
    
    # 步骤3: 应用全局剪枝，确保每层保留至少64个参数
    total_pruned = 0
    for module, (weight, max_prune) in layer_info.items():
        # 创建掩码
        abs_weights = torch.abs(weight)
        mask = abs_weights > global_threshold
        
        # 计算当前剪枝量
        current_prune = (mask == 0).sum().item()
        
        # 如果当前剪枝量超过该层最大允许值
        if current_prune > max_prune:
            # 计算该层实际阈值（确保保留至少64个参数）
            flat_weights = abs_weights.view(-1)
            sorted_weights = torch.sort(flat_weights).values
            layer_threshold = sorted_weights[-65]  # 第65大的值
            
            # 应用层级阈值
            mask = abs_weights > layer_threshold
            current_prune = (mask == 0).sum().item()
        
        # 应用剪枝
        weight.mul_(mask)
        total_pruned += current_prune
        
        # 存储掩码用于后续操作
        module.register_buffer('weight_mask', mask)
    
    # 计算实际稀疏度
    actual_sparsity = total_pruned / total_params
    print(f"Target sparsity: {target_sparsity:.2f}, Actual sparsity: {actual_sparsity:.2f}")
    print(f"Total parameters: {total_params}, Pruned: {total_pruned}")

def fuse_pruning(model):
    """
    将剪枝结果永久化 - 仅对支持的层进行
    Args:
        model: 剪枝后的模型
    """
    for name, module in model.named_modules():
        # 只处理QuantLinear和QuantConv2d层
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
            
        # 检查是否有存储的掩码
        if hasattr(module, 'weight_mask'):
            # 永久化剪枝结果
            mask = module.weight_mask
            module.weight.data.mul_(mask)
            
            # 移除临时存储的掩码
            delattr(module, 'weight_mask')





# # 辅助函数：冻结零权重
# def freeze_zero_weights(model):
#     """冻结模型中值为零的权重"""
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             # 创建掩码：非零位置为True，零位置为False
#             mask = param.data != 0
#             # 冻结零权重：将零权重位置的requires_grad设置为False
#             param.requires_grad = mask
#             print(f"冻结 {name} 中的零权重 ({torch.sum(~mask).item()} 个权重被冻结)")
