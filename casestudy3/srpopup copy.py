import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq
import math
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import copy
import random
from models import get_model
from dataset import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq
import math
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import copy
import random

class SRPopupTrainer:
    def __init__(self, model_name='2c3f', w=4, a=4, epochs=500, random_seed=1998, 
                 dataset_name='MNIST', batch_size=64, lr=0.01, optimizer='sgd', 
                 momentum=0.9, weight_decay=1e-4, lr_scheduler='plateau', patience=10, 
                 threshold=0.5, clip_weights=True, min_lr=1e-6, model=None, freeze_zeros=False,
                 prune_rate=0.9, prune_layers=None, eta=0.99, is_binary=False,
                 pretrained_path=None):
        """
        SRPopupTrainer 初始化
        
        参数:
            model_name: 模型名称 (默认 '2c3f')
            w: 权重位宽 (默认 4)
            a: 激活位宽 (默认 4)
            epochs: 训练轮数 (默认 500)
            random_seed: 随机种子 (默认 1998)
            dataset_name: 数据集名称 (默认 'MNIST')
            batch_size: 批大小 (默认 64)
            lr: 初始学习率 (默认 0.01)
            optimizer: 优化器类型 ('adam' 或 'sgd') (默认 'sgd')
            momentum: SGD动量 (默认 0.9)
            weight_decay: 权重衰减 (默认 1e-4)
            lr_scheduler: 学习率调度器 ('plateau', 'step', 'cosine' 或 None) (默认 'plateau')
            patience: 早停/学习率降低的耐心值 (默认 10)
            threshold: 二分类阈值 (默认 0.5)
            clip_weights: 是否裁剪权重 (默认 True)
            min_lr: 最小学习率 (默认 1e-6)
            model: 预初始化的模型 (默认 None)
            freeze_zeros: 是否冻结零权重 (默认 False)
            prune_rate: 剪枝率 (默认 0.9)
            prune_layers: 指定要剪枝的层 (None表示所有层)
            eta: 剪枝权重的初始掩码值 (默认 0.99)
            is_binary: 是否为二分类任务 (默认 False)
            pretrained_path: 预训练权重路径 (默认 None)
        """
        # 保存所有参数
        self.model_name = model_name
        self.w = w
        self.a = a
        self.epochs = epochs
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_type = optimizer.lower()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler.lower() if lr_scheduler else None
        self.patience = patience
        self.threshold = threshold
        self.clip_weights = clip_weights
        self.min_lr = min_lr
        self.model = model
        self.freeze_zeros = freeze_zeros
        self.prune_rate = prune_rate
        self.prune_layers = prune_layers
        self.eta = eta
        self.is_binary = is_binary
        self.pretrained_path = pretrained_path
        
        # 设置随机种子
        self._set_seed()
        
        # 获取设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建目录
        self.build_dir = "./build"
        self.model_dir = "./model"
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 加载预训练权重（如果提供路径）
        if self.pretrained_path:
            self.load_pretrained(self.pretrained_path)
        
        # 准备数据加载器
        self._init_dataloaders()
        
        # 准备模型：添加掩码参数
        self.prepare_model()
        
        # 优化器：只优化掩码参数
        mask_params = [param for name, param in self.model.named_parameters() if 'mask' in name]
        self.optimizer = self._get_optimizer(mask_params)
        
        # 学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 损失函数 - 根据任务类型选择
        if self.is_binary:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.stopped_early = False
    
    def load_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        
        state_dict = torch.load(weight_path, map_location=self.device)
        model_state_dict = self.model.state_dict()
        
        # 更宽松的加载策略
        matched_state_dict = {}
        for k, v in state_dict.items():
            # 尝试直接匹配
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                matched_state_dict[k] = v
                continue
                
            # 尝试移除可能的模块前缀
            new_k = k.replace('module.', '')
            if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
                matched_state_dict[new_k] = v
                continue
                
            # 尝试添加量化后缀 (量化模型可能有的额外参数)
            quant_k = k + '.weight'
            if quant_k in model_state_dict and model_state_dict[quant_k].shape == v.shape:
                matched_state_dict[quant_k] = v
                continue
        
        # 加载匹配的权重
        model_state_dict.update(matched_state_dict)
        self.model.load_state_dict(model_state_dict, strict=False)
        print(f"从 {weight_path} 加载预训练权重, 匹配了 {len(matched_state_dict)}/{len(state_dict)} 个参数")

    
    def _set_seed(self):
        """设置全局随机种子"""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_model(self):
        """初始化模型"""
        if self.model is None:
            # 如果没有提供预初始化的模型，则创建新模型
            self.model = get_model(self.model_name, self.w, self.a)
            print(f"创建新模型: {self.model_name}")
        else:
            # 使用提供的模型
            print(f"使用预初始化的模型: {self.model_name}")
        
        self.model.to(self.device)
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size
        )
    
    def _get_optimizer(self, params):
        """创建优化器"""
        if self.optimizer_type == 'adam':
            return optim.Adam(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            return optim.SGD(
                params, 
                lr=self.lr, 
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.optimizer_type}")
    
    def _get_scheduler(self):
        """创建学习率调度器"""
        if not self.lr_scheduler_type:
            return None
            
        if self.lr_scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5,
                patience=self.patience // 2,
                verbose=True,
                min_lr=self.min_lr
            )
        elif self.lr_scheduler_type == 'step':
            return StepLR(
                self.optimizer, 
                step_size=self.epochs // 5, 
                gamma=0.1
            )
        elif self.lr_scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(f"不支持的学习率调度器: {self.lr_scheduler_type}")
    
    def prepare_model(self):
        """
        准备模型：为指定层添加掩码参数
        """
        # 如果未指定剪枝层，默认剪枝所有全连接层和卷积层
        if self.prune_layers is None:
            self.prune_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.prune_layers.append((name, module))
        else:
            # 将层名称映射到模块对象
            pruned_modules = []
            for name in self.prune_layers:
                module = self.model
                for part in name.split('.'):
                    module = getattr(module, part)
                pruned_modules.append((name, module))
            self.prune_layers = pruned_modules
        
        # 添加掩码参数并初始化
        for name, module in self.prune_layers:
            # 为模块添加掩码参数
            # 使用安全的参数名称，避免点号
            param_name = f"mask_{name.replace('.', '_')}"
            
            # 如果已经添加过掩码，跳过
            if hasattr(module, param_name):
                continue
                
            # 添加掩码参数
            mask = torch.ones_like(module.weight.data, requires_grad=True)
            module.register_parameter(param_name, nn.Parameter(mask))
            
            # 冻结权重（只优化掩码）
            module.weight.requires_grad = False
        
        # 使用幅度剪枝初始化掩码
        self.initialize_mask_with_magnitude()
            # 添加STE函数
        class BinaryMaskSTE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, mask_param, prune_rate, eta):
                # 二值化掩码
                all_mask_vals = mask_param.view(-1)
                k = int(all_mask_vals.numel() * prune_rate)
                threshold = torch.topk(all_mask_vals, k, largest=False).values[-1]
                binary_mask = (mask_param > threshold).float()
                ctx.save_for_backward(mask_param, binary_mask)
                return binary_mask

            @staticmethod
            def backward(ctx, grad_output):
                mask_param, binary_mask = ctx.saved_tensors
                # 直通估计器: 通过二值化操作的梯度
                grad_input = grad_output.clone()
                # 可选: 添加梯度裁剪
                grad_input = torch.clamp(grad_input, -1.0, 1.0)
                return grad_input, None, None
        
        self.binary_mask_ste = BinaryMaskSTE.apply
    
    def initialize_mask_with_magnitude(self):
        """
        使用幅度剪枝初始化掩码
        """
        # 收集所有权重
        all_weights = []
        for name, module in self.prune_layers:
            all_weights.append(module.weight.data.abs().view(-1))
        
        all_weights = torch.cat(all_weights)
        k = int(all_weights.numel() * self.prune_rate)
        threshold = torch.topk(all_weights, k, largest=False).values[-1]
        
        # 应用幅度剪枝初始化掩码
        for name, module in self.prune_layers:
            # 获取掩码参数
            param_name = f"mask_{name.replace('.', '_')}"
            mask_param = getattr(module, param_name)
            
            weight_abs = module.weight.data.abs()
            
            # 初始化掩码：保留的权重为1，剪枝的权重为eta
            init_mask = torch.ones_like(mask_param.data)
            init_mask[weight_abs < threshold] = self.eta
            mask_param.data = init_mask
    
    def apply_mask(self):
        """应用带STE的掩码到权重"""
        for name, module in self.prune_layers:
            param_name = f"mask_{name.replace('.', '_')}"
            mask_param = getattr(module, param_name)
            
            # 使用STE获得可导的二值掩码
            binary_mask = self.binary_mask_ste(mask_param, self.prune_rate, self.eta)
            
            # 应用二值化掩码 - 注意: 不修改权重本身，只返回掩码后的权重
            return module.weight * binary_mask
    
    def apply_mask_for_layer(self, name, module):
        """为单个层应用掩码"""
        param_name = f"mask_{name.replace('.', '_')}"
        mask_param = getattr(module, param_name)
        
        # 使用STE
        binary_mask = self.binary_mask_ste(mask_param, self.prune_rate, self.eta)
        
        # 返回掩码后的权重（不修改原始权重）
        return module.weight * binary_mask

    def forward(self, x):
        """带掩码的前向传播（不修改原始权重）"""
        # 保存原始权重
        original_weights = {}
        for name, module in self.prune_layers:
            original_weights[name] = module.weight.data.clone()
            
            # 临时应用掩码
            masked_weight = self.apply_mask_for_layer(name, module)
            module.weight.data = masked_weight
        
        # 前向传播
        output = self.model(x)
        
        # 恢复原始权重
        for name, module in self.prune_layers:
            module.weight.data = original_weights[name]

        return output
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 计算权重交换数量
        total_iterations = len(self.train_loader)
        current_iteration = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            # ...
            # 前向传播 (现在在forward内部处理掩码)
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 应用短限制弹出 (保持原样)
            self.apply_short_restriction(epoch, batch_idx, total_iterations)
            
            # 更新掩码参数
            self.optimizer.step()
            # 计算准确率
            if self.is_binary:
                predicted = (torch.sigmoid(outputs) > self.threshold).float()
                correct += (predicted == targets).sum().item()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            # 统计损失
            total_loss += loss.item()
            total += targets.size(0)
            current_iteration += 1
        
        # 更新学习率
        if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()
        
        # 计算指标
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    
    def apply_short_restriction(self, epoch, batch_idx, total_iterations):
        """
        应用短限制弹出策略（完整实现）
        """
        # 计算当前迭代的权重交换率
        t = (epoch - 1) * total_iterations + batch_idx
        t_f = self.epochs * total_iterations
        rate = math.ceil((1 - t / t_f) ** 4)
        
        # 创建临时掩码副本用于交换
        temp_masks = {}
        for name, module in self.prune_layers:
            param_name = f"mask_{name.replace('.', '_')}"
            mask_param = getattr(module, param_name)
            temp_masks[name] = mask_param.data.clone()
        
        # 收集所有掩码值
        all_masks = []
        all_names = []
        all_indices = []
        
        # 遍历所有剪枝层
        for name, module in self.prune_layers:
            mask_data = temp_masks[name]  # 使用临时副本
            
            # 展平掩码并收集
            flat_mask = mask_data.view(-1)
            all_masks.append(flat_mask)
            
            # 记录名称和索引
            num_elements = mask_data.numel()
            all_names.extend([name] * num_elements)
            
            # 创建索引映射
            if len(mask_data.shape) == 1:
                indices = [(i,) for i in range(mask_data.size(0))]
            elif len(mask_data.shape) == 2:
                indices = [(i, j) for i in range(mask_data.size(0)) for j in range(mask_data.size(1))]
            elif len(mask_data.shape) == 4:
                indices = [(i, j, k, l) for i in range(mask_data.size(0)) 
                        for j in range(mask_data.size(1)) 
                        for k in range(mask_data.size(2)) 
                        for l in range(mask_data.size(3))]
            all_indices.extend(indices)
        
        if not all_masks:
            return
        
        # 合并所有掩码值
        all_masks = torch.cat(all_masks)
        
        # 计算当前剪枝率（如果使用渐进式剪枝）
        current_prune_rate = self.current_prune_rate if hasattr(self, 'current_prune_rate') else self.prune_rate
        
        # 计算二值化阈值
        k = int(all_masks.numel() * current_prune_rate)
        if k == 0:
            return  # 没有需要剪枝的元素
        
        # 找到阈值（最小的前k个值中的最大值）
        threshold = torch.kthvalue(all_masks, k).values
        
        # 识别保留和剪枝的权重
        preserved_mask = all_masks > threshold
        pruned_mask = all_masks <= threshold
        
        preserved_indices = torch.nonzero(preserved_mask, as_tuple=False).squeeze()
        pruned_indices = torch.nonzero(pruned_mask, as_tuple=False).squeeze()
        
        # 确保索引是1D张量
        if preserved_indices.dim() == 0:
            preserved_indices = preserved_indices.unsqueeze(0)
        if pruned_indices.dim() == 0:
            pruned_indices = pruned_indices.unsqueeze(0)
        
        # 限制权重交换数量
        q_t = min(rate, pruned_indices.size(0), preserved_indices.size(0))
        
        if q_t > 0:
            # 选择要交换的权重
            # 剪枝部分中最大的q_t个值（最可能恢复）
            _, top_pruned = torch.topk(all_masks[pruned_indices], q_t, largest=True)
            # 保留部分中最小的q_t个值（最可能剪枝）
            _, bottom_preserved = torch.topk(all_masks[preserved_indices], q_t, largest=False)
            
            # 交换掩码值
            for idx in top_pruned:
                name = all_names[pruned_indices[idx]]
                indices = all_indices[pruned_indices[idx]]
                
                # 获取临时掩码
                mask_data = temp_masks[name]
                
                # 更新掩码值（恢复权重）
                if len(indices) == 1:
                    mask_data[indices[0]] = 1.0
                elif len(indices) == 2:
                    mask_data[indices[0], indices[1]] = 1.0
                elif len(indices) == 4:
                    mask_data[indices[0], indices[1], indices[2], indices[3]] = 1.0
            
            for idx in bottom_preserved:
                name = all_names[preserved_indices[idx]]
                indices = all_indices[preserved_indices[idx]]
                
                # 获取临时掩码
                mask_data = temp_masks[name]
                
                # 更新掩码值（剪枝权重）
                if len(indices) == 1:
                    mask_data[indices[0]] = self.eta
                elif len(indices) == 2:
                    mask_data[indices[0], indices[1]] = self.eta
                elif len(indices) == 4:
                    mask_data[indices[0], indices[1], indices[2], indices[3]] = self.eta
        
        # 应用交换后的掩码值到模型参数
        for name, module in self.prune_layers:
            param_name = f"mask_{name.replace('.', '_')}"
            mask_param = getattr(module, param_name)
            mask_param.data = temp_masks[name]
    
    def validate(self):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 处理二分类任务的标签
                if self.is_binary:
                    targets = targets.float().view(-1, 1)
                
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                
                # 计算准确率
                if self.is_binary:
                    predicted = (torch.sigmoid(outputs) > self.threshold).float()
                    correct += (predicted == targets).sum().item()
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                
                total_loss += loss.item()
                total += targets.size(0)
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def test(self):
        """
        测试模型
        """
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.forward(inputs)
                
                if self.is_binary:
                    # 处理二分类预测
                    preds = (torch.sigmoid(outputs) > self.threshold).int()
                    targets = targets.view(-1, 1).int()
                else:
                    # 处理多分类预测
                    _, preds = outputs.max(1)
                
                y_true.extend(targets.cpu().numpy().flatten())
                y_pred.extend(preds.cpu().numpy().flatten())
        
        test_acc = 100 * accuracy_score(y_true, y_pred)
        return test_acc
    
    def finalize_model(self):
        """
        最终化模型：应用掩码并移除掩码参数
        """
        # 应用最终掩码
        self.apply_mask()
        
        # 移除掩码参数
        for name, module in self.prune_layers:
            # 构造安全的参数名称
            param_name = f"mask_{name.replace('.', '_')}"
            
            if hasattr(module, param_name):
                # 移除掩码参数
                delattr(module, param_name)
                
                # 解冻权重
                module.weight.requires_grad = True
                
                # 如果设置了冻结零权重，冻结权重值为零的位置
                if self.freeze_zeros:
                    mask = (module.weight.data != 0)
                    module.weight.requires_grad = mask
    
    def train_model(self):
        """
        训练模型主函数
        """
        # 准备日志文件
        logger_path = os.path.join(
            self.model_dir,
            f"{self.model_name}_w{self.w}_a{self.a}_e{self.epochs}.log"
        )
        
        with open(logger_path, 'w') as log_file:
            # 记录配置信息
            log_file.write(f"训练配置:\n")
            log_file.write(f"模型: {self.model_name}, 权重位宽: {self.w}, 激活位宽: {self.a}\n")
            log_file.write(f"数据集: {self.dataset_name}, 批大小: {self.batch_size}, 训练轮数: {self.epochs}\n")
            log_file.write(f"优化器: {self.optimizer_type}, 学习率: {self.lr}, 权重衰减: {self.weight_decay}\n")
            log_file.write(f"学习率调度器: {self.lr_scheduler_type}, 耐心值: {self.patience}, 最小学习率: {self.min_lr}\n")
            log_file.write(f"二分类阈值: {self.threshold}, 权重裁剪: {self.clip_weights}\n")
            log_file.write(f"冻结零权重: {self.freeze_zeros}, 剪枝率: {self.prune_rate}\n")
            log_file.write(f"剪枝层: {[name for name, _ in self.prune_layers] if self.prune_layers else '所有层'}\n")
            if self.pretrained_path:
                log_file.write(f"预训练权重: {self.pretrained_path}\n")
            log_file.write("\n")
            
            log_file.write(f"{'Epoch':^6} | {'LR':^10} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | Status\n")
            log_file.write("-" * 85 + "\n")
            
            best_model = None
            best_val_acc = 0.0
            best_epoch = 0
            
            for epoch in range(1, self.epochs + 1):
                if self.stopped_early:
                    log_file.write(f"⏹ 在第 {epoch} 轮提前停止\n")
                    print(f"⏹ 在第 {epoch} 轮提前停止")
                    break
                
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_acc = self.validate()
                
                # 更新学习率 (针对ReduceLROnPlateau)
                current_lr = self.optimizer.param_groups[0]['lr']
                status = ""
                
                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    # 检测学习率是否变化
                    if new_lr < current_lr:
                        status = "🔽 LR Reduced"
                        self.epochs_no_improve = 0  # 重置早停计数器
                        current_lr = new_lr
                
                # 更新最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.epochs_no_improve = 0
                    status = "★ Best" if not status else status + " ★ Best"
                else:
                    self.epochs_no_improve += 1
                
                # 早停检查
                if self.epochs_no_improve >= self.patience:
                    if current_lr <= self.min_lr:
                        self.stopped_early = True
                        status = "⏹ Early Stop"
                    else:
                        # 如果还有学习空间，只重置早停计数器
                        self.epochs_no_improve = 0
                
                # 记录日志
                log_msg = (
                    f"{epoch:^6} | {current_lr:^10.7f} | "
                    f"{train_loss:^10.4f} | {train_acc:^10.2f}% | "
                    f"{val_loss:^10.4f} | {val_acc:^10.2f}% | {status}"
                )
                print(log_msg)
                log_file.write(log_msg + '\n')
            
            # 加载最佳模型
            if best_model:
                self.model.load_state_dict(best_model)
                log_file.write(f"\n最佳验证准确率: {best_val_acc:.2f}% (第 {best_epoch} 轮)\n")
                print(f"\n最佳验证准确率: {best_val_acc:.2f}% (第 {best_epoch} 轮)")
            
            # 最终化模型
            self.finalize_model()
            
            # 测试
            test_acc = self.test()
            log_file.write(f"测试准确率: {test_acc:.2f}%\n")
            print(f"测试准确率: {test_acc:.2f}%")
            
            # 保存最终模型
            model_save_path = os.path.join(
                self.model_dir, 
                f"{self.model_name}_w{self.w}_a{self.a}_e{self.epochs}.pth"
            )
            torch.save(self.model.state_dict(), model_save_path)
            log_file.write(f"最终模型保存至: {model_save_path}\n")
            print(f"最终模型保存至: {model_save_path}")
        
        return self.model
if __name__ == "__main__":


    trainer = SRPopupTrainer(
        model_name='2c3f_relu',
        w=4,
        a=4,
        epochs=500,
        random_seed=1998,
        dataset_name='MNIST',
        batch_size=64,
        lr=0.01,
        optimizer='sgd',
        momentum=0.9,
        weight_decay=1e-4,
        lr_scheduler='plateau',
        patience=10,
        threshold=0.5,
        clip_weights=True,
        min_lr=1e-6,
        model=None,  # 自动创建模型
        freeze_zeros=False,
        prune_rate=0.1,
        prune_layers=None,  # 剪枝所有层
        eta=0.99,
        is_binary=False,  # 多分类任务
        pretrained_path='pretrained/2c3f_relu_w4_a4_pretrained.pth'  # 预训练权重路径
    )

    # 训练模型
    pruned_model = trainer.train_model()