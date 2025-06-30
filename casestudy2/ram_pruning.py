from models import get_model
import torch
from torch import nn
from build_ram import build_ram, draw_bipartite_adjacency_graph
import os
import numpy as np
from pruning import analyze_model_sparsity
import torch.nn.utils.prune as prune
if __name__ == "__main__":
    # Get the model
    model_name = "tfc"
    w = 1
    a = 1
    weight_path = '/home/changhong/prj/finn/script/EF_US/casestudy1/model/best_tfc_w1_a1_100.pth'
    model = get_model(model_name, w, a)
    model.load_state_dict(torch.load(weight_path))
    show_graph = False

    mask_dir = './mask_tmp'
    os.makedirs(mask_dir, exist_ok=True)  # 正确创建目录

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
                module.weight.data = module.weight.data * mask
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

    