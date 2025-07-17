import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
import onnx
import os
from sklearn.metrics import accuracy_score
from models import get_model
from dataset import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

build_dir = "./build"
model_dir = "./model"
data_dir = "./data"

os.makedirs(build_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
print(f"Data directory: {data_dir}")
print(f"Build directory: {build_dir}")
print(f"Model directory: {model_dir}")

class Trainer:
    def __init__(self, model_name='2c3f', w=4, a=4, epochs=500, random_seed=1998, dataset_name='MNIST'):
        self.model_name = model_name
        self.w = w
        self.a = a
        self.epochs = epochs
        self.random_seed = random_seed
        self.dataset_name = dataset_name

    def train(self, model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        threshold = 0.5
        
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

    def validate(self, model, loader, criterion, device, threshold=0.5):

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        is_binary = isinstance(criterion, nn.BCEWithLogitsLoss)
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if is_binary:
                    labels = labels.float().view(-1, 1)
                    loss = criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > threshold).float()
                    correct += (predicted == labels).sum().item()
                else:
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


    def test(self, model, test_loader, device, threshold=0.5):

        model.to(device)  # 确保模型在正确的设备上
        model.eval()
        y_true = []
        y_pred = []

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
                    probs = torch.sigmoid(outputs)
                    pred = (probs > threshold).int().cpu().numpy()
                    target = target.view(-1, 1).int().cpu().numpy()
                else:
                    _, pred = torch.max(outputs, 1)
                    if target.dim() > 1: 
                        _, target = torch.max(target, 1)
                    pred = pred.cpu().numpy()
                    target = target.cpu().numpy()
                
                y_true.extend(target.flatten().tolist())
                y_pred.extend(pred.flatten().tolist())

        return 100 * accuracy_score(y_true, y_pred)



    def train_model(self):#), model_name='2c3f', w=8, a=8, epochs=500, random_seed=1998, dataset_name='MNIST'):
        random_seed = self.random_seed
        model_name = self.model_name
        w = self.w
        a = self.a
        epochs = self.epochs
        dataset_name = self.dataset_name

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)


        model = get_model(model_name, w, a)
        model.to(device)
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name=dataset_name)

        if model_name == 'unsw_fc':
            criterion = nn.BCEWithLogitsLoss()  # For binary classification tasks like UNSW-NB15
        else:
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
            train_loss, train_acc = self.train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = self.validate(model, val_loader, criterion, device)
            
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
        test_acc = self.test(model, test_loader, device)

        log(f'Test Accuracy of the best model on the test images: {test_acc:.2f}%')
        torch.save(model.state_dict(), model_save_path)
 
        log(f'Model saved to {model_save_path}')
        return  model