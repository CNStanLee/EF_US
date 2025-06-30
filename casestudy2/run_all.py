from models import get_model
import os
import torch
from train import train_try
from pruning import auto_prune_model
from generate import generate_ip
import copy
from estimate import estimate_ip
from auto_unfold import auto_unfold_json
import time
def run_all(model_name = '2c3f', w = 4, a = 4, epochs = 500, random_seed = 1998, dataset_name = 'MNIST', pruning_type = 'l1', tryid = "test", folding_config_file="unfold"):
    # Stage 1: init ---------------------------------------------------------

    # directories
    # no change
    try_name=f"{model_name}_w{w}_a{a}_{epochs}_{folding_config_file}_{tryid}"
    try_name="/" + try_name + '_' + pruning_type
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folding_json_file = script_dir + f'/estimates_output{try_name}/auto_folding_config.json' 
    onnx_model_file = script_dir + f'/estimates_output{try_name}/intermediate_models/step_apply_folding_config.onnx'
    # unfold_json_file =  script_dir + f'/estimates_output{try_name}/unfold.json'
    unfold_json = script_dir + f'/estimates_output{try_name}/unfold.json'
    auto_json = script_dir + f'/estimates_output{try_name}/auto.json'
    log_file = script_dir + f'/build{try_name}/log.txt'
    os.makedirs(script_dir + f'/build{try_name}', exist_ok=True)
    # init log file
    def log(message):
        with open(log_file, 'a') as f:
            f.write(message + '\n')
            print(message)
    start_date_time = time.localtime()
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', start_date_time)}")
    
    # set start date time

    log('Stage 1: Initializing the environment and parameters')
    # Stage 2: get model
    log('Stage 2: Getting the model')
    model = get_model(model_name, w, a)
    # Stage 3: train model
    log('Stage 3: Training the model')
    # if the model pth exists, load it
    model_weight_name = f"./model/best_{model_name}_w{w}_a{a}_{epochs}.pth"
    # if the model weight exists, load it
    if os.path.exists(model_weight_name):
        log(f"Loading model weight from {model_weight_name}")
        model.load_state_dict(torch.load(model_weight_name))
    else:
        log(f"Model weight {model_weight_name} does not exist, training the model")
        model = train_try(model_name = model_name, w = w, a = a, epochs = epochs, random_seed = random_seed)
    # Stage 4: prune model
    if pruning_type == 'na':
        pruned_model = copy.deepcopy(model)
    else:
        log('Stage 4: Pruning the model')
        model_prune_weight_name = f"./model/final_{model_name}_w{w}_a{a}_{pruning_type}_pruned.pth"
        if os.path.exists(model_prune_weight_name):
            log(f"Loading model weight from {model_prune_weight_name}")
            model.load_state_dict(torch.load(model_prune_weight_name))
            pruned_model = copy.deepcopy(model)
        else:
            pruned_model = auto_prune_model(model, model_name, w, a, dataset_name = dataset_name, pruning_type= pruning_type)
    # Stage 5: estimate the model
    log('Stage 5: Estimating the model')
    estimate_ip(model_name=model_name, model = pruned_model, weight=w, activation=a, try_name=try_name)
    auto_unfold_json(folding_json_file = folding_json_file, onnx_model_file = onnx_model_file, unfold_json = unfold_json, auto_json = auto_json)
    # Stage 6: generate ip
    log('Stage 6: Generating the IP')

    if folding_config_file == "auto":
        generate_ip(model_name=model_name, model = pruned_model, weight=w, activation=a, try_name=try_name, folding_config_file= auto_json)
    elif folding_config_file == "unfold":
        generate_ip(model_name=model_name, model = pruned_model, weight=w, activation=a, try_name=try_name, folding_config_file= unfold_json)
    else:
        generate_ip(model_name=model_name, model = pruned_model, weight=w, activation=a, try_name=try_name, folding_config_file = folding_config_file)
    end_date_time = time.localtime()
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', end_date_time)}")
    # total time consumed xxday, xxhour, xxmin, xxsec
    total_time = time.mktime(end_date_time) - time.mktime(start_date_time)
    days = total_time // (24 * 3600)
    total_time %= (24 * 3600)
    hours = total_time // 3600
    total_time %= 3600
    minutes = total_time // 60
    seconds = total_time % 60
    log(f"Total time consumed: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
    log('All stages completed successfully!')
if __name__ == "__main__":
    # run_all(model_name = 'tfc',
    #          w = 1,
    #            a = 1, 
    #            epochs = 500,
    #              random_seed = 1998,
    #                dataset_name = 'MNIST',
    #                  pruning_type = 'na',
    #                    tryid = "test2",
    #                      folding_config_file="auto")
    
    # run_all(model_name = 'tfc',
    #         w = 1,
    #         a = 1, 
    #         epochs = 500,
    #             random_seed = 1998,
    #             dataset_name = 'MNIST',
    #                 pruning_type = 'ram',
    #                 tryid = "test2",
    #                     folding_config_file="auto")

    # run_all(model_name = 'tfc',
    #     w = 1,
    #     a = 1, 
    #     epochs = 500,
    #         random_seed = 1998,
    #         dataset_name = 'MNIST',
    #             pruning_type = 'ram',
    #             tryid = "test2",
    #                 folding_config_file="unfold")
    
    # run_all(model_name = '2c3f',
    #     w = 4,
    #     a = 4, 
    #     epochs = 500,
    #         random_seed = 1998,
    #         dataset_name = 'MNIST',
    #             pruning_type = 'l1',
    #             tryid = "test2",
    #                 folding_config_file="auto")
    
    # run_all(model_name = '2c3f',
    #     w = 4,
    #     a = 4, 
    #     epochs = 500,
    #         random_seed = 1998,
    #         dataset_name = 'MNIST',
    #             pruning_type = 'l1',
    #             tryid = "test2",
    #                 folding_config_file="unfold")
    
    # run_all(model_name = '2c3f',
    #     w = 4,
    #     a = 4, 
    #     epochs = 500,
    #         random_seed = 1998,
    #         dataset_name = 'MNIST',
    #             pruning_type = 'na',
    #             tryid = "test2",
    #                 folding_config_file="unfold")
    run_all(model_name = 'tfc',
    w = 1,
    a = 1, 
    epochs = 500,
        random_seed = 1998,
        dataset_name = 'MNIST',
            pruning_type = 'na',
            tryid = "test3",
                folding_config_file="unfold")
    
    run_all(model_name = '2c3f',
    w = 1,
    a = 1, 
    epochs = 500,
        random_seed = 1998,
        dataset_name = 'MNIST',
            pruning_type = 'na',
            tryid = "test3",
                folding_config_file="auto")