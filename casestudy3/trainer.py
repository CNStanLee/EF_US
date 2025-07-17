import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models import get_model
from dataset import get_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 目录配置
build_dir = "./build"
model_dir = "./model"
data_dir = "./data"

os.makedirs(build_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

class Trainer:
    def __init__(self, model_name='2c3f', w=4, a=4, epochs=500, random_seed=1998, dataset_name='MNIST',
                 batch_size=64, lr=0.01, optimizer='sgd', momentum=0.9, weight_decay=1e-4,
                 lr_scheduler='plateau', patience=10, threshold=0.5, clip_weights=True, min_lr=1e-6,
                 model=None, freeze_zeros=False):
        """
        初始化训练器
        
        参数:
            model_name: 模型名称 (默认 '2c3f')
            w: 权重位宽 (默认 4)
            a: 激活位宽 (默认 4)
            epochs: 训练轮数 (默认 500)
            random_seed: 随机种子 (默认 1998)
            dataset_name: 数据集名称 (默认 'MNIST')
            batch_size: 批大小 (默认 64)
            lr: 初始学习率 (默认 0.01)
            optimizer: 优化器类型 ('adam' 或 'sgd') (默认 'adam')
            momentum: SGD动量 (默认 0.9)
            weight_decay: 权重衰减 (默认 1e-4)
            lr_scheduler: 学习率调度器 ('plateau', 'step', 'cosine' 或 None) (默认 'step')
            patience: 早停/学习率降低的耐心值 (默认 10)
            threshold: 二分类阈值 (默认 0.5)
            clip_weights: 是否裁剪权重 (默认 True)
            min_lr: 最小学习率 (默认 1e-6)
            model: 预初始化的模型 (默认 None)
            freeze_zeros: 是否冻结零权重 (默认 False)
        """
        # 基本参数
        self.model_name = model_name
        self.w = w
        self.a = a
        self.epochs = epochs
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        
        # 训练超参数
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
        
        # 模型相关
        self.model = model
        self.freeze_zeros = freeze_zeros
        
        # 统一任务类型判断标准
        self.is_binary = (model_name == 'unsw_fc')
        
        # 训练状态
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.stopped_early = False

    def _set_seed(self):
        """设置全局随机种子"""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_pretrained(self, model, weight_path):
        """
        从指定路径加载预训练权重
        
        参数:
            model: 要加载权重的模型
            weight_path: 预训练权重文件路径
            
        返回:
            加载了预训练权重的模型
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
            
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"loading {weight_path} to weights")
        return model

    def freeze_zero_weights(self, model):
        """冻结模型中值为零的权重"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                # 创建掩码：非零位置为True，零位置为False
                mask = param.data != 0
                # 冻结零权重：将零权重位置的requires_grad设置为False
                param.requires_grad = mask
                print(f"冻结 {name} 中的零权重 ({torch.sum(~mask).item()} 个权重被冻结)")

    def train(self, model, loader, criterion, optimizer, device):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            if self.is_binary:
                # 二分类处理
                labels = labels.float().view(-1, 1)
                loss = criterion(outputs, labels)
                predicted = (torch.sigmoid(outputs) > self.threshold).float()
                correct += (predicted == labels).sum().item()
            else:
                # 多分类处理
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            
            loss.backward()
            
            # 在优化器步骤之前应用梯度掩码（如果冻结了零权重）
            if self.freeze_zeros:
                for name, param in model.named_parameters():
                    if 'weight' in name and hasattr(param, 'requires_grad_mask'):
                        # 确保梯度只在非零位置更新
                        if param.grad is not None:
                            param.grad *= param.requires_grad_mask.float()
            
            optimizer.step()
            
            # 权重裁剪
            if self.clip_weights and hasattr(model, 'clip_weights'):
                model.clip_weights(-1.0, 1.0)
            
            running_loss += loss.item()
            total += labels.size(0)
        
        train_loss = running_loss / len(loader)
        train_acc = 100 * correct / total
        return train_loss, train_acc

    def validate(self, model, loader, criterion, device):
        """验证模型性能"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if self.is_binary:
                    labels = labels.float().view(-1, 1)
                    loss = criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > self.threshold).float()
                    correct += (predicted == labels).sum().item()
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                
                running_loss += loss.item()
                total += labels.size(0)
        
        val_loss = running_loss / len(loader)
        val_acc = 100 * correct / total
        return val_loss, val_acc

    def test(self, model, test_loader, device):
        """
        测试模型性能
        
        参数:
            model: 要测试的模型
            test_loader: 测试数据加载器
            device: 使用的设备
            
        返回:
            测试准确率 (百分比)
        """
        model.eval()
        model.to(device)  # 确保整个模型在目标设备上
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing"):
                # 确保数据和模型在同一设备上
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                if self.is_binary:
                    # 使用 .detach().cpu() 确保计算在 CPU 上进行
                    preds = (torch.sigmoid(outputs) > self.threshold).int().detach().cpu().numpy()
                    targets = targets.view(-1, 1).int().detach().cpu().numpy()
                else:
                    _, preds = torch.max(outputs, 1)
                    preds = preds.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                
                y_true.extend(targets.flatten().tolist())
                y_pred.extend(preds.flatten().tolist())
        
        test_acc = 100 * accuracy_score(y_true, y_pred)
        return test_acc

    def test_model(self, model):
        """
        独立测试预训练模型
        
        参数:
            model: 要测试的模型
            
        返回:
            测试准确率 (百分比)
        """
        
        # 获取数据加载器
        _, _, test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size
        )
        
        # 执行测试
        return self.test(model, test_loader, device)

    def _get_optimizer(self, model):
        """创建优化器"""
        # 只优化需要梯度的参数
        params = [p for p in model.parameters() if p.requires_grad]
        
        if not params:
            raise ValueError("没有可优化的参数！请检查模型参数是否被正确设置。")
        
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

    def _get_scheduler(self, optimizer):
        """创建学习率调度器"""
        if not self.lr_scheduler_type:
            return None
            
        if self.lr_scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=0.5,
                patience=self.patience // 2,
                verbose=True,
                min_lr=self.min_lr
            )
        elif self.lr_scheduler_type == 'step':
            return StepLR(
                optimizer, 
                step_size=self.epochs // 5, 
                gamma=0.1
            )
        elif self.lr_scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer, 
                T_max=self.epochs,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(f"不支持的学习率调度器: {self.lr_scheduler_type}")

    def train_model(self):
        """训练模型主函数"""
        # 设置随机种子
        self._set_seed()
        
        # 初始化模型
        if self.model is None:
            # 如果没有提供预初始化的模型，则创建新模型
            model = get_model(self.model_name, self.w, self.a)
            print(f"创建新模型: {self.model_name}")
        else:
            # 使用提供的模型
            model = self.model
            print(f"使用预初始化的模型: {self.model_name}")
        
        model.to(device)
        
        # 冻结零权重（如果需要）
        if self.freeze_zeros:
            self.freeze_zero_weights(model)
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size
        )
        
        # 选择损失函数
        criterion = nn.BCEWithLogitsLoss() if self.is_binary else nn.CrossEntropyLoss()
        
        # 创建优化器和学习率调度器
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        
        # 准备模型保存路径
        model_save_path = os.path.join(
            model_dir, 
            f"{self.model_name}_w{self.w}_a{self.a}_e{self.epochs}.pth"
        )
        best_model_save_path = os.path.join(
            model_dir, 
            f"best_{self.model_name}_w{self.w}_a{self.a}_e{self.epochs}.pth"
        )
        logger_path = os.path.join(
            model_dir,
            f"{self.model_name}_w{self.w}_a{self.a}_e{self.epochs}.log"
        )
        
        # 训练循环
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.stopped_early = False
        
        with open(logger_path, 'w') as log_file:
            # 记录配置信息
            log_file.write(f"训练配置:\n")
            log_file.write(f"模型: {self.model_name}, 权重位宽: {self.w}, 激活位宽: {self.a}\n")
            log_file.write(f"数据集: {self.dataset_name}, 批大小: {self.batch_size}, 训练轮数: {self.epochs}\n")
            log_file.write(f"优化器: {self.optimizer_type}, 学习率: {self.lr}, 权重衰减: {self.weight_decay}\n")
            log_file.write(f"学习率调度器: {self.lr_scheduler_type}, 耐心值: {self.patience}, 最小学习率: {self.min_lr}\n")
            log_file.write(f"二分类阈值: {self.threshold}, 权重裁剪: {self.clip_weights}\n")
            log_file.write(f"冻结零权重: {self.freeze_zeros}\n\n")
            
            log_file.write(f"{'Epoch':^6} | {'LR':^10} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | Status\n")
            log_file.write("-" * 85 + "\n")
            
            for epoch in range(self.epochs):
                if self.stopped_early:
                    log_file.write(f"⏹ 在第 {epoch} 轮提前停止\n")
                    print(f"⏹ 在第 {epoch} 轮提前停止")
                    break
                
                # 训练和验证
                train_loss, train_acc = self.train(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc = self.validate(
                    model, val_loader, criterion, device
                )
                
                # 更新学习率
                current_lr = optimizer.param_groups[0]['lr']
                status = ""
                
                # 更新最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    torch.save(model.state_dict(), best_model_save_path)
                    self.epochs_no_improve = 0
                    status = "★ Best"
                else:
                    self.epochs_no_improve += 1
                
                # 学习率调度（仅针对Plateau）
                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)  # 传入监控指标
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    # 检测学习率是否变化
                    if new_lr < current_lr:
                        status = "🔽 LR Reduced"
                        self.epochs_no_improve = 0  # 重置早停计数器
                        current_lr = new_lr
                
                # 其他类型学习率调度
                elif scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                
                # 早停检查（独立于学习率调度）
                if self.epochs_no_improve >= self.patience:
                    if current_lr <= self.min_lr:
                        self.stopped_early = True
                        status = "⏹ Early Stop"
                    else:
                        # 如果还有学习空间，只重置早停计数器
                        self.epochs_no_improve = 0
                
                # 记录日志
                log_msg = (
                    f"{epoch+1:^6} | {current_lr:^10.7f} | "
                    f"{train_loss:^10.4f} | {train_acc:^10.2f}% | "
                    f"{val_loss:^10.4f} | {val_acc:^10.2f}% | {status}"
                )
                print(log_msg)
                log_file.write(log_msg + '\n')
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(best_model_save_path))
        test_acc = self.test(model, test_loader, device)
        
        # 保存最终模型
        torch.save(model.state_dict(), model_save_path)
        
        # 最终日志
        final_msg = (
            f"\n训练完成: 最佳验证准确率 {self.best_val_acc:.2f}% (第 {self.best_epoch+1} 轮)\n"
            f"测试准确率: {test_acc:.2f}%\n"
            f"最佳模型保存至: {best_model_save_path}\n"
            f"最终模型保存至: {model_save_path}"
        )
        
        print(final_msg)
        with open(logger_path, 'a') as log_file:
            log_file.write(final_msg + '\n')
        
        return model
