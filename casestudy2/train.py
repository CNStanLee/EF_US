import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
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


from models import get_model
from dataset import get_dataloaders

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# work directory setup 
# notebook_name = "/EF-US-Engine-Free-Unstructured-Sparsity-Design-Alleviates-Accelerator-Bottlenecks/casestudy1/LeNet_MNIST_BNN"
# finn_root = os.getcwd()
build_dir = "./build"
model_dir = "./model"
data_dir = "./data"

# Create directories if they do not exist
os.makedirs(build_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
print(f"Data directory: {data_dir}")
print(f"Build directory: {build_dir}")
print(f"Model directory: {model_dir}")


def train(model, loader, criterion, optimizer, device, threshold=0.5):
    """
    修改后的训练函数（与test函数逻辑一致）
    参数:
        model: 待训练模型
        loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        threshold: 二分类阈值（默认0.5）
    返回:
        训练损失和准确率（百分比）
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 统一使用损失函数类型判断任务类型
    is_binary = isinstance(criterion, nn.BCEWithLogitsLoss)
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        if is_binary:
            # 统一形状处理：(N, 1)
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
            
            # 使用与test函数相同的预测逻辑
            predicted = (torch.sigmoid(outputs) > threshold).float()
            correct += (predicted == labels).sum().item()

            
        else:
            # 多分类处理
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            if labels.dim() > 1:  # 处理one-hot标签
                _, labels = torch.max(labels, 1)
            correct += (predicted == labels).sum().item()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
    
    train_loss = running_loss / len(loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(model, loader, criterion, device, threshold=0.5):
    """
    修改后的验证函数（与test函数逻辑一致）
    参数:
        model: 待验证模型
        loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        threshold: 二分类阈值（默认0.5）
    返回:
        验证损失和准确率（百分比）
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 统一使用损失函数类型判断任务类型
    is_binary = isinstance(criterion, nn.BCEWithLogitsLoss)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if is_binary:
                # 统一形状处理：(N, 1)
                labels = labels.float().view(-1, 1)
                loss = criterion(outputs, labels)
                
                # 使用与test函数相同的预测逻辑
                predicted = (torch.sigmoid(outputs) > threshold).float()
                correct += (predicted == labels).sum().item()
            else:
                # 多分类处理
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                if labels.dim() > 1:  # 处理one-hot标签
                    _, labels = torch.max(labels, 1)
                correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            total += labels.size(0)
    
    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

from sklearn.metrics import accuracy_score
def test(model, test_loader, device, threshold=0.5):
    """
    通用测试函数（支持二分类/多分类自动判断）
    参数:
        model: 待测试模型
        test_loader: 测试数据加载器
        device: 计算设备
        threshold: 二分类阈值（默认0.5）
    返回:
        测试准确率（百分比）
    """
    import torch
    from sklearn.metrics import accuracy_score

    model.to(device)  # 确保模型在正确的设备上
    model.eval()
    y_true = []
    y_pred = []

    # 自动检测任务类型（通过首次前向传播）
    with torch.no_grad():
        sample_input, _ = next(iter(test_loader))
        sample_input = sample_input.to(device)
        output_shape = model(sample_input[0:1]).shape
        is_binary = output_shape[-1] == 1  # 二分类输出维度为1

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            
            if is_binary:
                # 二分类处理
                probs = torch.sigmoid(outputs)
                pred = (probs > threshold).int().cpu().numpy()
                target = target.view(-1, 1).int().cpu().numpy()
            else:
                # 多分类处理
                _, pred = torch.max(outputs, 1)
                if target.dim() > 1:  # 处理one-hot标签
                    _, target = torch.max(target, 1)
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
            
            y_true.extend(target.flatten().tolist())
            y_pred.extend(pred.flatten().tolist())

    return 100 * accuracy_score(y_true, y_pred)



def train_try(model_name='2c3f', w=8, a=8, epochs=500, random_seed=1998, dataset_name='MNIST'):
    
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


    model = get_model(model_name, w, a)
    model.to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name=dataset_name)

    # Define loss function and optimizer

    if model_name == 'unsw_fc':
        criterion = nn.BCEWithLogitsLoss()  # For binary classification tasks like UNSW-NB15
    else:
        # For classification tasks like MNIST, CIFAR-10, etc.
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs/5, gamma=0.1)

    # train temp parameters
    best_val_acc = 0.0
    model_save_path = model_dir + f"/{model_name}_w{w}_a{a}_{epochs}.pth"
    best_model_save_path = model_dir + f"/best_{model_name}_w{w}_a{a}_{epochs}.pth"
    logger_path = model_dir + f"/{model_name}_w{w}_a{a}_{epochs}.log"

    # define a logger
    if not os.path.exists(logger_path):
        with open(logger_path, 'w') as f:
            f.write(f"Training {model_name} with weight bit width {w} and activation bit width {a} for {epochs} epochs.\n")

    def log(message):
        with open(logger_path, 'a') as f:
            f.write(message + '\n')
        print(message)

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if hasattr(model, 'clip_weights'):
            model.clip_weights(-1.0, 1.0)
        current_lr = optimizer.param_groups[0]['lr']
        
        log(f'Epoch [{epoch+1}/{epochs}], '
            f'LR: {current_lr:.6f}, '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_save_path)

    model.load_state_dict(torch.load(best_model_save_path))
    model.to(device)
    test_acc = test(model, test_loader, device)
    #print(f'Test Accuracy of the best model on the test images: {test_acc:.2f}%')
    log(f'Test Accuracy of the best model on the test images: {test_acc:.2f}%')
    torch.save(model.state_dict(), model_save_path)
    #print(f'Model saved to {model_save_path}')
    log(f'Model saved to {model_save_path}')
    return  model#, model_name, model_save_path


def main():

    model_name = 'unsw_fc'
    w = 2
    a = 2
    model = get_model(model_name, w, a)
    epochs = 1000
    model.to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name='UNSW')
    # load model weight
    model_weight_name = f"./model/best_{model_name}_w{w}_a{a}_{epochs}.pth"
    if os.path.exists(model_weight_name):
        print(f"Loading model weight from {model_weight_name}")
        model.load_state_dict(torch.load(model_weight_name))
    else:
        print(f"Model weight {model_weight_name} does not exist, training the model")
        model = train_try(model_name=model_name, w=w, a=a, epochs=epochs, random_seed=1998, dataset_name='UNSW')
    # test the model
    test_acc = test(model, test_loader, device)
    print(f'Test Accuracy of the model on the test images: {test_acc:.2f}%')
    # save the model



if __name__ == "__main__":
    main()