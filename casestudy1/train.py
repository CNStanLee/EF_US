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


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    return test_acc

def train_try(model_name='2c3f', w=8, a=8, epochs=500, random_seed=1998):
    
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


    model = get_model(model_name, w, a)
    model.to(device)
    train_loader, val_loader, test_loader = get_dataloaders()

    # Define loss function and optimizer
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
    #train_try(model_name='2c3f', w=8, a=8, epochs=10, random_seed=1998)

    w_set = [8]
    a_set = [8]
    epochs = 500

    for w in w_set:
        for a in a_set:
            print(f"Training model with weight bit width {w} and activation bit width {a}")
            train_try(model_name='2c3f', w=w, a=a, epochs=epochs, random_seed=1998)
            print(f"Finished training model with weight bit width {w} and activation bit width {a}")

if __name__ == "__main__":
    main()