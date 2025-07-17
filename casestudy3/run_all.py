from dataset import get_dataloaders
from utils.logger import Logger
from models import get_model
from trainer import Trainer
import yaml
import torch
from pruning import analyze_model_sparsity, fuse_pruning, global_magnitude_prune_with_min, freeze_zero_weights
from finn_estimate import estimate_ip
from bn_estimator import resource_analysis, cycle_analysis, get_layer_channels, get_onnx_model, modify_mvau_parallelization, unfold_node, get_node_names
import onnx
import brevitas.nn as qnn
def main():
    # ------ paras ------
    divider = "=" * 50
    config_file = "config/unsw/95sparsity30porch.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    try_id = config.get('try_id', "0715")

    model_name = config.get('arch', 'unsw_fc')
    dataset_name = config.get('data_set', 'UNSW')
    w = config.get('w', 2)
    a = config.get('a', 2)
    training_epochs = config.get('training_epochs', 500)
    random_seed = config.get('random_seed', 1998)
    pruning_type = config.get('pruning_type', 'l1')
    folding_config_file = config.get('folding_config_file', "auto")
    pretrained_model = config.get('pretrained_model', None)
    optimizer = config.get('optimizer', 'adam')
    lr = config.get('lr', 0.001)
    lr_scheduler = config.get('lr_scheduler', 'step')
    training_weight_decay = config.get('training_weight_decay', 1e-4) # convert to float
    if isinstance(training_weight_decay, str):
        training_weight_decay = float(training_weight_decay)
    patience = config.get('patience', 25)
    min_lr = config.get('min_lr', 1e-7)
    if isinstance(min_lr, str):
        min_lr = float(min_lr)
    momentum = config.get('momentum', 0.9)
    training_batch_size = config.get('training_batch_size', 256)
    binary_threshold = config.get('binary_threshold', 0.5)
    retrain_epochs = config.get('retrain_epochs', 1)
    # ------ paras ------
    logger = Logger("casestudy3.log")
    logger.log(divider)
    logger.log_start_time()
    logger.log("Starting the run_all process...")
    logger.log(f"Model Name: {model_name}, Width: {w}, Activation: {a}, Epochs: {training_epochs}, "
               f"Random Seed: {random_seed}, Dataset Name: {dataset_name}, "
               f"Pruning Type: {pruning_type}, Try ID: {try_id}, Folding Config File: {folding_config_file}")

    # Step 1: Model training or loading
    logger.log(divider)
    logger.log("Step 1: Model training or loading...")
    trainer = Trainer(
        model_name = model_name,  
        w = w,            
        a = a,                
        epochs = training_epochs,         
        batch_size=training_batch_size,     
        lr=lr,           
        optimizer=optimizer,   
        weight_decay=training_weight_decay,  
        lr_scheduler=lr_scheduler,  
        patience=patience,       
        min_lr=min_lr,        
        momentum=momentum,     
        threshold=binary_threshold,     
        clip_weights=True,   
        dataset_name=dataset_name,  
        random_seed=random_seed
    )

    if pretrained_model is not None:
        logger.log(f"Loading pretrained model from {pretrained_model}")
        model = get_model(model_name, w, a)
        
        # pre_layer = qnn.QuantIdentity(bit_width=a)
        # model = torch.nn.Sequential(pre_layer, model)
        # # save the weight to new pth file
        # torch.save(model.state_dict(), pretrained_model)
        trained_model = trainer.load_pretrained(model, pretrained_model)   
    else:
        logger.log("No pretrained model provided, using the trained model.")
        trained_model = trainer.train_model()

    test_acc = trainer.test_model(trained_model)
    logger.log(f"Test accuracy: {test_acc:.2f}%")
    # Step 2: Model pruning estimation
    logger.log(divider)
    logger.log("Step 2: Model pruning estimation...")
    # 1. 创建稀疏决策
    # pruning_decisions = make_sparsity_decision(model, target_sparsity=0.95)
    # 2. 执行预剪枝
    global_magnitude_prune_with_min(model, target_sparsity=0.98)
    # 3. 融合剪枝结果
    fuse_pruning(model)
    # 4. 分析剪枝后模型
    analyze_model_sparsity(model)
    freeze_zero_weights(model)
    pruned_model = model
    retrainer = Trainer(
        model_name = model_name,  
        w = w,            
        a = a,                
        epochs = retrain_epochs,         
        batch_size=training_batch_size,     
        lr=lr,           
        optimizer=optimizer,   
        weight_decay=training_weight_decay,  
        lr_scheduler=lr_scheduler,  
        patience=patience,       
        min_lr=min_lr,        
        momentum=momentum,     
        threshold=binary_threshold,     
        clip_weights=True,   
        dataset_name=dataset_name,  
        model=pruned_model,
        random_seed=random_seed
    )
    retrained_model = retrainer.train_model()
    test_acc = retrainer.test_model(retrained_model)
    print(f"测试准确率: {test_acc:.2f}%")
    analyze_model_sparsity(retrained_model)

    # Step 3: Evaluate the resoucrce consumption and bottle neck to make prune decisions
    logger.log(divider)
    logger.log("Step 3: Evaluate the resource consumption and bottle neck to make prune decisions...")

    pre_layer = qnn.QuantIdentity(bit_width=a)
    retrained_model = torch.nn.Sequential(pre_layer, retrained_model)

    estimate_ip(model_name = model_name,
                 model = retrained_model,
                   weight=w,
                     activation=a,
                       try_name=try_id)


    model_path = f"./estimates_output/{model_name}_{w}_{a}_{try_id}/intermediate_models/step_generate_estimate_reports.onnx"
    model = get_onnx_model(model_path)
    onnx_model = onnx.load(model_path)
    channel_info = get_layer_channels(onnx_model)


    print(divider)
    print('Analyzing auto fold model')
    print(divider)
    auto_cycle_result = cycle_analysis(model)
    auto_res_result = resource_analysis(model, fpgapart = "xcu50-fsvh2104-2L-e")
    print("\n\n")
    node_names = get_node_names(model)
    for node_name in node_names:
        modify_mvau_parallelization(model, node_name, pe=1, simd=1)
    print(divider)
    print('Analyzing auto fully folded model')
    print(divider)
    fold_cycle_result = cycle_analysis(model, plot=True)
    fold_res_result = resource_analysis(model, fpgapart="xcu50-fsvh2104-2L-e", plot=True)
    print("\n\n")
    for node_name in node_names:
        unfold_node(model, node_name, channel_info)
    print(divider)
    print('Analyzing auto fully unfolded model')
    print(divider)
    unfold_cycle_result = cycle_analysis(model, plot=True)
    unfold_res_result = resource_analysis(model, fpgapart="xcu50-fsvh2104-2L-e", plot=True)
    print("\n\n")

    # Step 6: Prune the model with LTH approach

    # Step 7: generate the IP

    # Step 8: Bechmarking the IP

if __name__ == "__main__":
    main()