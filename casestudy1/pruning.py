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
# from finn.util.test import get_test_model_trained
# from brevitas.export import export_qonnx
# from qonnx.util.cleanup import cleanup as qonnx_cleanup
# from qonnx.core.modelwrapper import ModelWrapper
# from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
# from qonnx.transformation.infer_shapes import InferShapes
# from qonnx.transformation.fold_constants import FoldConstants
# from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
# from finn.util.basic import make_build_dir
# from finn.util.visualization import showInNetron
from dataclasses import dataclass
import copy
import torch.nn.utils.prune as prune
import numpy as np
from collections import namedtuple, defaultdict


from dataset import get_dataloaders
from models import get_model
from train import train, test, validate
from build_ram import build_ram, draw_bipartite_adjacency_graph

import json
import csv
from collections import namedtuple

# 保存结果到CSV
def save_to_csv(results, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(results[0]._fields)
        # 写入数据
        for res in results:
            writer.writerow([res.layer_path, res.param_count ,res.pruning_percentage, 
                            res.test_accuracy, res.accuray_drop])

# 从CSV读取结果
def load_from_csv(filename):
    PruningResult = namedtuple('PruningResult', ['layer_path', 'param_count','pruning_percentage', 'test_accuracy', 'accuray_drop'])
    results = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            # 转换数据类型
            layer_path = row[0]
            param_count = float(row[1])  # 添加参数计数
            pruning_percentage = float(row[2])
            test_accuracy = float(row[3])
            accuray_drop = float(row[4])
            results.append(PruningResult(
                layer_path, param_count, pruning_percentage, test_accuracy, accuray_drop
            ))
    return results

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

def layer_pruning(model, param_path, pruning_percentage=0.5, pruning_type='l1'):
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
    
    if pruning_type == 'l1':
        prune.l1_unstructured(target_module, name=param_name, amount=pruning_percentage)
        prune.remove(target_module, param_name)
    elif pruning_type == 'l2':
        prune.l2_unstructured(target_module, name=param_name, amount=pruning_percentage)
        prune.remove(target_module, param_name)
    elif pruning_type == 'random':
        prune.random_unstructured(target_module, name=param_name, amount=pruning_percentage)
        prune.remove(target_module, param_name)
    elif pruning_type == 'magnitude':
        prune.magnitude_unstructured(target_module, name=param_name, amount=pruning_percentage)
        prune.remove(target_module, param_name)
    elif pruning_type == 'SNIP':
        # SNIP pruning is not directly supported in PyTorch's prune module.
        # You would need to implement SNIP pruning logic separately.
        raise NotImplementedError("SNIP pruning is not implemented in this function.")
    else:
        raise ValueError(f"Unsupported pruning type: {pruning_type}")
    
    return model_copy



def sensitivity_analysis(model, test_loader, sparsity_info, pruning_type='l1', step = 20):
    # 定义命名元组类型
    PruningResult = namedtuple('PruningResult', ['layer_path', 'param_count','pruning_percentage', 'test_accuracy', 'accuray_drop'])
    
    original_acc = test(model, test_loader, device)
    results = []
    for param in sparsity_info:
        layer_path = param.path
        param_count = param.param_count
        for percentage in range(0, 100, step):
            percentage /= 100.0
            # Apply pruning
            pruned_model = layer_pruning(model, layer_path, pruning_percentage=percentage, pruning_type=pruning_type)
            test_acc = test(pruned_model, test_loader, device)
            
            # 创建命名元组并添加到结果列表
            result = PruningResult(
                layer_path=layer_path,
                param_count=param_count,
                pruning_percentage=percentage,
                test_accuracy=test_acc,
                accuray_drop= original_acc - test_acc
            )
            results.append(result)
            
            print(f"Layer {layer_path} Pruning {percentage*100:.0f}% - Test Accuracy: {test_acc:.2f}%, Accuracy Drop: {result.accuray_drop:.2f}%")
    return results


from collections import namedtuple

def determine_safe_pruning_rates(sensitivity_results, accuracy_drop_tolerance=0.05, threshold=200):
    """
    根据敏感度分析结果确定每层的安全剪枝率
    
    参数:
    sensitivity_results -- sensitivity_analysis()函数的返回结果列表
    accuracy_drop_tolerance -- 可接受的精度下降阈值 (默认0.05)
    
    返回:
    PruningDecision命名元组列表，包含每层路径和安全剪枝率
    """
    # 定义结果命名元组
    PruningDecision = namedtuple('PruningDecision', ['layer_path', 'pruning_rate'])
    
    # 按层路径分组结果
    layer_results = {}
    for result in sensitivity_results:
        layer_path = result.layer_path
        if layer_path not in layer_results:
            layer_results[layer_path] = []
        layer_results[layer_path].append(result)
    
    decisions = []
    # 处理每层数据
    for layer_path, results in layer_results.items():
        # 按剪枝率降序排序 (从高剪枝率到低剪枝率)
        sorted_results = sorted(results, key=lambda x: x.pruning_percentage, reverse=True)
        
        safe_rate = 0.0  # 默认安全剪枝率为0
        # 从高剪枝率向低剪枝率搜索
        for result in sorted_results:
            if result.accuray_drop <= accuracy_drop_tolerance:
                safe_rate = result.pruning_percentage
                break  # 找到第一个满足条件的即停止

        # when layer parameter count is too small, skip pruning
        if results[0].param_count < threshold:
            safe_rate = 0.0
        
        decisions.append(PruningDecision(layer_path, safe_rate))
    
    return decisions


def prune_model_by_decisions(original_model, pruning_decisions, pruning_type='l1'):
    """
    根据剪枝决策列表对模型进行逐层剪枝
    
    参数:
        original_model: 原始模型
        pruning_decisions: 剪枝决策列表，每个元素是PruningDecision（包含layer_path和pruning_rate）
        pruning_type: 剪枝类型，默认为'l1'
    
    返回:
        剪枝后的模型
    """
    # 创建一个模型副本用于剪枝
    pruned_model = copy.deepcopy(original_model)
    pruned_model.to(device)
    
    # 对每一层应用剪枝
    for decision in pruning_decisions:
        # 应用剪枝
        pruned_model = layer_pruning(
            model=pruned_model,
            param_path=decision.layer_path,
            pruning_percentage=decision.pruning_rate,
            pruning_type=pruning_type
        )
    
    return pruned_model

def freeze_zero_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            mask = (param != 0).float()
            param.register_hook(lambda grad, mask=mask: grad * mask)

def retrain_model(model, train_loader, val_loader, epochs=20):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs/5, gamma=0.1)

    freeze_zero_weights(model)  
    best_val_acc = 0.0
    best_model = copy.deepcopy(model)
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if hasattr(model, 'clip_weights'):
            model.clip_weights(-1.0, 1.0)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{epochs}], '
            f'LR: {current_lr:.6f}, '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
    return best_model






# pruning method:
# l1
# l2
# random
# magnitude
# Ram
# SNIP

def main():
    # input parameters
    w = 4
    a = 4
    model_name = "2c3f_relu"
    model_weight = "./model/best_2c3f_relu_w4_a4_500.pth"

    # Analyze model sparsity
    print("Analyzing model sparsity...")
    ori_model = get_model(model_name, w, a)
    ori_model.load_state_dict(torch.load(model_weight))
    ori_model.to(device)

    sparsity_info = analyze_model_sparsity(ori_model)

    # 'path',        # 完整参数路径 (如 "conv1.weight")
    # 'layer_type',  # 层类型 (如 "Conv2d")
    # 'param_count', # 参数总数
    # 'non_zero',    # 非零参数数量
    # 'sparsity'     # 稀疏度百分比

    # 遍历所有参数

    # deep copy the model
    # pmodel = copy.deepcopy(model).to(device)

    # for param in sparsity_info:
    #     if param.param_count > 150: 
    #         pmodel = apply_l1_pruning_by_param_path(pmodel, param.path, pruning_percentage=0.5, verbose=True)

    # sparsity_info = analyze_model_sparsity(pmodel)

    # lock random seed
    torch.manual_seed(1998)
    torch.cuda.manual_seed(1998)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    model = copy.deepcopy(ori_model).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name='MNIST')
    # Stage 1: sensitivity analysis
    print("\nStage 1: Sensitivity Analysis")
    # prune each layer seprately from 10% to 90% with step 10% and check accuracy decrease
    # original_acc = test(model, test_loader, device)

    original_acc = 0
    for i in range(10):
        original_acc_1 = test(model, test_loader, device)
        original_acc += original_acc_1
        #print(original_acc_1)
    original_acc /= 10
    print(f"Original Test Accuracy: {original_acc:.2f}%")
    
    sensitivity_result_name = model_name + '_sensitivity_results.csv'
    if not os.path.exists(sensitivity_result_name):
        sensitivity_results = sensitivity_analysis(model, test_loader, sparsity_info, pruning_type='l1', step = 1)
        # save
        save_to_csv(sensitivity_results, model_name +'_sensitivity_results.csv')
    else:
        # read
        sensitivity_results = load_from_csv(model_name +'_sensitivity_results.csv')
    # Stage 2: With a tolerance, prune each layer with their specific pruning rate
    converge = False
    print("\nStage 2: Pruning with Specific Rates")
    accuracy_drop_tolerance = 83 # 5% accuracy drop tolerance
    final_accuracy_drop_tolerance = 30  # 1% final accuracy drop tolerance
    final_model = copy.deepcopy(model)
    while not converge:
        pruning_decisions = determine_safe_pruning_rates(sensitivity_results, accuracy_drop_tolerance)
        # Print results
        for decision in pruning_decisions:
            print(f"Layer {decision.layer_path} - Safe pruning rate: {decision.pruning_rate*100:.1f}%")

        pruned_model = prune_model_by_decisions(model, pruning_decisions, pruning_type='l1')
        analyze_model_sparsity(pruned_model)
        # Stage 3: Re-train the model, try to fit accuracy dropdown acceptable
        print("\nStage 3: Re-training the Pruned Model")
        # retrained_model = retrain_model(model, train_loader, val_loader, epochs=5)
        # retrained_model = retrain_model(pruned_model, train_loader, num_epochs=5)
        retrained_model = retrain_model(pruned_model, train_loader, val_loader, epochs=200)
        
        analyze_model_sparsity(retrained_model)
        
        # Stage 4: if accuracy is not acceptable, try to change tolerance and pruning rate, do 4 again
        # find 10 times average accuracy
        test_acc = 0
        for i in range(10):
            test_acc_1 = test(retrained_model, test_loader, device)
            test_acc += test_acc_1
        test_acc /= 10

        
        print(f"Test Accuracy after re-training: {test_acc:.2f}%")
        accuracy_drop = original_acc - test_acc
        print(f"Accuracy drop after pruning and re-training: {accuracy_drop:.2f}%")
        if accuracy_drop <= final_accuracy_drop_tolerance:
            print("Final model is acceptable, try bigger accuracy_drop_tolerance")
            accuracy_drop_tolerance += 2
            final_model = copy.deepcopy(retrained_model)
        else:
            # Increase the tolerance for the next iteration
            converge = True
            print(f"Increasing accuracy drop tolerance to {accuracy_drop_tolerance}% for next iteration.")

    # Stage 5: export the model to ONNX and FINN format
    print("\nStage 5: Exporting the Final Model")
    sparsity_info = analyze_model_sparsity(final_model)
    test_acc = test(final_model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    final_model = retrain_model(final_model, train_loader, val_loader, epochs=200)
    test_acc = test(final_model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    analyze_model_sparsity(final_model)

    # save model to pth
    model_save_path = f"./model/final_{model_name}_w{w}_a{a}_pruned.pth"
    torch.save(final_model.state_dict(), model_save_path)

    # check the total number of parameters
    total_params = sum(info.param_count for info in sparsity_info)
    total_non_zero = sum(info.non_zero for info in sparsity_info)
    print(f"Total Parameters: {total_params:,}, Non-zero Parameters: {total_non_zero:,}")
    compression_ratio = total_params / total_non_zero if total_non_zero > 0 else float('inf')
    compression_ratio = compression_ratio * 32 / (w)  # assuming 32-bit to w-bit weight and a-bit activation quantization
    print(f"Compression Ratio (pruning only): {compression_ratio:.2f}x")

def test_pruned_model():


    w = 4
    a = 4
    model_name = "2c3f_relu"
    model_weight = "./model/best_2c3f_relu_w4_a4_500.pth"
    # load the model
    model = get_model(model_name, w, a)
    model.load_state_dict(torch.load(model_weight))
    model.to(device)

    # get the test dataloader
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name='MNIST')

    pruned_model = layer_pruning(
            model=model,
            param_path='linear_features.0.weight',
            pruning_percentage=0.95,
            pruning_type='l1'
        )
    pruned_model = layer_pruning(
            model=pruned_model,
            param_path='conv_features.9.weight',
            pruning_percentage=0.9,
            pruning_type='l1'
        )
    # test the model
    test_acc = test(pruned_model, test_loader, device)
    print(f"Test Accuracy of the pruned model: {test_acc:.2f}%")
    retrained_model = retrain_model(pruned_model, train_loader, val_loader, epochs=100)
    test_acc = test(retrained_model, test_loader, device)
    print(f"Test Accuracy of the pruned model: {test_acc:.2f}%")
    analyze_model_sparsity(retrained_model)

def auto_ram_pruning(model ,show_graph = False, model_name= '2c3f', w = 4, a = 4, pruning_type = 'ram'):
    mask_dir = './mask_tmp'
    os.makedirs(mask_dir, exist_ok=True)  
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            print(f"Layer: {name}")
            print(f"Weight shape: {module.weight.shape}")
            if module.weight.dim() == 2 or module.weight.dim() == 4:
                print("This is a 2D or 4D weight tensor.")
                graph_shape = module.weight.shape

                safe_name = name.replace('.', '_')
                mask_file = os.path.join(mask_dir, f"mask_{model_name}_{safe_name}.npy")

                if not os.path.exists(mask_file):
                    print(f"Mask file {mask_file} does not exist, building the mask.")
                    mask = build_ram(graph_shape)  
                    np.save(mask_file, mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
                else:
                    print(f"Mask file {mask_file} exists, loading the mask.")
                    mask = torch.tensor(np.load(mask_file)) 

                # if input channel and output channel both < 100 draw the bipartite adjacency graph
                if show_graph:
                    if graph_shape[0] < 100 and graph_shape[1] < 100:
                        print("Drawing the bipartite adjacency graph.")
                        draw_bipartite_adjacency_graph(mask)
                print(mask)

                print("-" * 50)
                print(f"Weight shape: {module.weight.shape}")
                print(module.weight.data)
                # use the mask to prune the weight tensor
                # get the type of the weight data
                weight_type = module.weight.data.dtype
                module.weight.data = module.weight.data * mask
                # convert the weight data to the same type as before
                module.weight.data = module.weight.data.to(weight_type)
                print("-" * 50)
                print(f"Pruned weight shape: {module.weight.shape}")
                # print the weight pruned tensor
                print(module.weight.data)
            else:
                print("This is an unsupported weight tensor dimension.")
            # print(module.weight.data)
            print("-" * 50)

    print("Finished processing all layers.")
    analyze_model_sparsity(model)
    model_save_path = f"./model/final_{model_name}_w{w}_a{a}_{pruning_type}_pruned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

def auto_prune_model(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1'):
    if pruning_type == 'ram':
        print('ram pruning')
        model = auto_ram_pruning(model = ori_model, show_graph = False, model_name = model_name, w = w, a = a, pruning_type = pruning_type)
    else:
        print('pruning with sens')
        model = auto_prune_model_sensitivity(ori_model = ori_model, model_name = model_name, w = w, a = a, dataset_name=dataset_name, pruning_type = pruning_type)
    return model

def auto_prune_model_sensitivity(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1'):

    accuracy_drop_tolerance = 30  # 40% layer accuracy drop tolerance
    final_accuracy_drop_tolerance = 10  # 1% final accuracy drop tolerance
    retrain_epochs = 10  # number of epochs to retrain the model after pruning
    final_retrain_epochs = 100  # number of epochs to retrain the final model after pruning
    sensitivity_step = 1    # step size for sensitivity analysis, 1% for each step
    tolerance_step = 1  # step size for increasing accuracy drop tolerance
    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    # ori_model = get_model(model_name, w, a)
    # ori_model.load_state_dict(torch.load(model_weight))
    ori_model.to(device) 

    sparsity_info = analyze_model_sparsity(ori_model)

    # init
    torch.manual_seed(1998)
    torch.cuda.manual_seed(1998)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    model = copy.deepcopy(ori_model).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name)
    # Stage 1: sensitivity analysis
    print("\nPruning Stage 1: Sensitivity Analysis")

    original_acc = 0
    for i in range(10):
        original_acc_1 = test(model, test_loader, device)
        original_acc += original_acc_1
    original_acc /= 10
    print(f"Original Test Accuracy: {original_acc:.2f}%")
    
    sensitivity_result_name = model_name + '_sensitivity_results.csv'
    if not os.path.exists(sensitivity_result_name):
        sensitivity_results = sensitivity_analysis(model, test_loader, sparsity_info, pruning_type=pruning_type, step = sensitivity_step)
        save_to_csv(sensitivity_results, model_name +'_sensitivity_results.csv')
    else:
        sensitivity_results = load_from_csv(model_name +'_sensitivity_results.csv')
    # Stage 2: With a tolerance, prune each layer with their specific pruning rate
    converge = False
    print("\nPruning Stage 2: Pruning with Specific Rates")
    # final_model = copy.deepcopy(model)
    got_final_model = False
    while not converge:
        pruning_decisions = determine_safe_pruning_rates(sensitivity_results, accuracy_drop_tolerance)
        # Print results
        for decision in pruning_decisions:
            print(f"Layer {decision.layer_path} - Safe pruning rate: {decision.pruning_rate*100:.1f}%")

        pruned_model = prune_model_by_decisions(model, pruning_decisions, pruning_type=pruning_type)
        # Stage 3: Re-train the model, try to fit accuracy dropdown acceptable
        print("\nPruning Stage 3: Re-training the Pruned Model")
        retrained_model = retrain_model(pruned_model, train_loader, val_loader, epochs=retrain_epochs)
        analyze_model_sparsity(retrained_model)
        # Stage 4: if accuracy is not acceptable, try to change tolerance and pruning rate, do 4 again
        # find 10 times average accuracy
        test_acc = 0
        for i in range(10):
            test_acc_1 = test(retrained_model, test_loader, device)
            test_acc += test_acc_1
        test_acc /= 10

        #final_model = copy.deepcopy(retrained_model)
        print(f"Test Accuracy after re-training: {test_acc:.2f}%")
        accuracy_drop = original_acc - test_acc
        print(f"Accuracy drop after pruning and re-training: {accuracy_drop:.2f}%")
        if accuracy_drop <= final_accuracy_drop_tolerance:
            print("Final model is acceptable, try bigger accuracy_drop_tolerance")
            print(f"Current layer accuracy dropdown tolerance: {accuracy_drop_tolerance}" )
            accuracy_drop_tolerance += tolerance_step
            final_model = copy.deepcopy(retrained_model)
            got_final_model = True
        else:
            # Increase the tolerance for the next iteration
            converge = True
            if not got_final_model:
                print('first try not converge')
                final_model = copy.deepcopy(retrained_model)
            print(f"end.")
        # final_model = copy.deepcopy(retrained_model)
    test_acc = test(final_model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    # Stage 5: export the model to ONNX and FINN format
    print("\nPruning Stage 5: Final Retraining")
    final_model2 = retrain_model(final_model, train_loader, val_loader, epochs=final_retrain_epochs)
    test_acc = test(final_model2, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    analyze_model_sparsity(final_model2)
    # check the total number of parameters
    total_params = sum(info.param_count for info in sparsity_info)
    total_non_zero = sum(info.non_zero for info in sparsity_info)
    print(f"Total Parameters: {total_params:,}, Non-zero Parameters: {total_non_zero:,}")
    compression_ratio = total_params / total_non_zero if total_non_zero > 0 else float('inf')
    compression_ratio = compression_ratio * 32 / (w)  # assuming 32-bit to w-bit weight and a-bit activation quantization
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print("\nPruning Stage 6: Exporting the Final Model")
    # save model to pth
    model_save_path = f"./model/final_{model_name}_w{w}_a{a}_{pruning_type}_pruned.pth"
    torch.save(final_model2.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return final_model2

 

if __name__ == "__main__":
    # main()
    # test_pruned_model()
    w = 4
    a = 4
    model_name = "2c3f_relu"
    model_weight = "./model/best_2c3f_relu_w4_a4_500.pth"

    # Analyze model sparsity
    print("Analyzing model sparsity...")
    ori_model = get_model(model_name, w, a)
    ori_model.load_state_dict(torch.load(model_weight))
    ori_model.to(device)
    auto_prune_model(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1')