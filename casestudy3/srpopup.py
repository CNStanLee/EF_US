from models import get_model
from trainer import Trainer
from utils.logger import Logger
from dataset import get_dataloaders
import torch
import torch.nn as nn
import math

def forward_with_mask(model, mask, inputs):
    # 应用二值化掩码
    def binarize(mask_val):
        return (mask_val > 0.5).float()  # STE二值化
    
    # 遍历所有模块应用掩码
    for name, module in model.named_modules():
        if name in mask:
            # 保存原始权重
            original_weight = module.weight.data.clone()
            # 应用二值化掩码
            module.weight.data = original_weight * binarize(mask[name])
    
    # 执行前向传播
    outputs = model(inputs)
    
    # 恢复原始权重（仅掩码被优化）
    for name, module in model.named_modules():
        if name in mask:
            module.weight.data = original_weight
    
    return outputs


def initialize_mask(model, prune_rate):
    mask = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # 获取权重张量
            weight = module.weight.data
            # 计算幅度重要性
            importance = weight.abs()
            # 计算需要保留的权重数量
            num_keep = int(weight.numel() * (1 - prune_rate))
            # 确定阈值
            threshold = torch.topk(importance.view(-1), num_keep, largest=True).values[-1]
            # 创建初始掩码（1表示保留，0.99表示可能移除）
            mask[name] = torch.where(importance >= threshold, 
                                    torch.ones_like(weight),
                                    torch.full_like(weight, 0.99))
    return mask

def sr_popup(model, mask, epoch, total_epochs):
    # 计算动态交换率 (1 - epoch/t)^4
    dynamic_rate = (1 - epoch / total_epochs) ** 4
    
    for name, module in model.named_modules():
        if name in mask:
            # 获取当前掩码
            current_mask = mask[name]
            # 分离保留和可能移除的索引
            preserved_idx = (current_mask == 1)
            pruned_idx = (current_mask < 1)
            
            # 计算当前层交换数量
            q_t = math.ceil(pruned_idx.sum().item() * dynamic_rate)
            
            # 在保留部分找最小值（可能移除的候选）
            preserved_values = current_mask[preserved_idx]
            min_preserved = torch.topk(preserved_values, q_t, largest=False).values
            
            # 在可能移除部分找最大值（可能恢复的候选）
            pruned_values = current_mask[pruned_idx]
            max_pruned = torch.topk(pruned_values, q_t, largest=True).values
            
            # 执行交换：更新掩码值
            if min_preserved.numel() > 0 and max_pruned.numel() > 0:
                current_mask[preserved_idx] = torch.where(
                    preserved_values <= min_preserved[-1],
                    torch.tensor(0.99),  # 降为可能移除
                    preserved_values
                )
                
                current_mask[pruned_idx] = torch.where(
                    pruned_values >= max_pruned[-1],
                    torch.tensor(1.0),  # 升为保留
                    pruned_values
                )
    
    return mask

def train_jackpot(model, dataloader, epochs, prune_rate, loss_func=nn.CrossEntropyLoss()):
    # 初始化掩码
    mask = initialize_mask(model, prune_rate)
    # 设置优化器（仅优化掩码）
    optimizer = torch.optim.SGD(list(mask.values()), lr=0.1)
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            # 前向传播（应用掩码）
            outputs = forward_with_mask(model, mask, inputs)
            loss = loss_func(outputs, targets)
            
            # 反向传播（STE自动处理梯度）
            loss.backward()
            optimizer.step()
            
            # 应用SR-popup
            mask = sr_popup(model, mask, epoch, epochs)


if __name__ == "__main__":
    # w = 2
    # a = 2 
    # model_name = 'unsw_fc'
    # pretrained_model = 'pretrained/unsw_fc_w2_a2_pretrained.pth'
    # logger = Logger("casestudy3.log")
    # logger.log("Testing SRPopup...")
    # trainer = Trainer(
    #     model_name=model_name,
    #     w=w,
    #     a=a,
    #     epochs=500,
    #     batch_size=256,
    #     lr=0.001,
    #     optimizer='adam',
    #     weight_decay=1e-4,
    #     lr_scheduler='step',
    #     patience=25,
    #     min_lr=1e-7,
    #     momentum=0.9,
    #     threshold=0.5,
    #     clip_weights=True,
    #     dataset_name='UNSW',
    # )
    # if pretrained_model is not None:
    #     logger.log(f"Loading pretrained model from {pretrained_model}")
    #     model = get_model(model_name, w, a)
    #     trained_model = trainer.load_pretrained(model, pretrained_model)   
    # else:
    #     logger.log("No pretrained model provided, using the trained model.")
    #     trained_model = trainer.train_model()

    # test_acc = trainer.test_model(trained_model)
    # logger.log(f"Test accuracy: {test_acc}")


    # train_loader, val_loader, test_loader = get_dataloaders('UNSW', batch_size=256)  # Load the dataloaders for UNSW dataset
    # prune_rate = 0.9  # Set the target sparsity level
    # epochs = 30

    # loss_func = nn.BCEWithLogitsLoss() if model_name == 'unsw_fc' else nn.CrossEntropyLoss()# BCELoss for binary classification

    # train_jackpot(model, train_loader, epochs, prune_rate, loss_func=loss_func)

    w = 4
    a = 4 
    model_name = '2c3f_relu'
    pretrained_model = 'pretrained/2c3f_relu_w4_a4_pretrained.pth'
    logger = Logger("casestudy3.log")
    logger.log("Testing SRPopup...")
    trainer = Trainer(
        model_name=model_name,
        w=w,
        a=a,
        epochs=500,
        batch_size=256,
        lr=0.001,
        optimizer='adam',
        weight_decay=1e-4,
        lr_scheduler='step',
        patience=25,
        min_lr=1e-7,
        momentum=0.9,
        threshold=0.5,
        clip_weights=True,
        dataset_name='MNIST',
    )
    if pretrained_model is not None:
        logger.log(f"Loading pretrained model from {pretrained_model}")
        model = get_model(model_name, w, a)
        trained_model = trainer.load_pretrained(model, pretrained_model)   
    else:
        logger.log("No pretrained model provided, using the trained model.")
        trained_model = trainer.train_model()

    test_acc = trainer.test_model(trained_model)
    logger.log(f"Test accuracy: {test_acc}")

    train_loader, val_loader, test_loader = get_dataloaders('MNIST', batch_size=256)  # Load the dataloaders for UNSW dataset
    prune_rate = 0.9  # Set the target sparsity level
    epochs = 30

    # loss_func = nn.BCEWithLogitsLoss() if model_name == 'unsw_fc' else nn.CrossEntropyLoss()# BCELoss for binary classification
    loss_func = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    train_jackpot(trained_model, train_loader, epochs, prune_rate, loss_func=loss_func)