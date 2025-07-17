from finn.util.visualization import showInNetron, showSrc
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from functools import partial
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import res_estimation
import matplotlib.pyplot as plt
from qonnx.custom_op.registry import getCustomOp
import onnx
from collections import defaultdict

pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC4x2"] = "xczu48dr-ffvg1517-2-e"
pynq_part_map["KV260_SOM"] = "xck26-sfvc784-2LV-c"
pynq_part_map["U50"] = "xcu50-fsvh2104-2L-e"
pynq_part_map["XCvu9P"] = "xcvu9p-flgb2104-2-i"

spliter = "----------------------------------------"

from collections import defaultdict

def get_layer_channels(onnx_model):
    """从ONNX模型中提取各层的输入输出通道数"""
    channel_info = defaultdict(dict)
    
    # 遍历所有节点
    for node in onnx_model.graph.node:
        layer_name = node.name
        
        # 获取输入张量信息
        for input_name in node.input:
            for tensor in onnx_model.graph.value_info:
                if tensor.name == input_name:
                    if tensor.type.tensor_type.HasField("shape"):
                        dims = tensor.type.tensor_type.shape.dim
                        if len(dims) > 1:  # 确保有通道维度
                            # if input_name contains 'out', we accept it as output
                            if 'out' in input_name:
                                channel_info[layer_name]["in_channels"] = dims[-1].dim_value
                            #else:
                            #channel_info[layer_name]["in_channels"] = dims[1].dim_value
                #else:
                    #channel_info[layer_name]["in_channels"] =  onnx_model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        
        # 获取输出张量信息
        for output_name in node.output:
            for tensor in onnx_model.graph.value_info:
                if tensor.name == output_name:
                    if tensor.type.tensor_type.HasField("shape"):
                        dims = tensor.type.tensor_type.shape.dim
                        if len(dims) > 1:  # 确保有通道维度
                            channel_info[layer_name]["out_channels"] = dims[-1].dim_value
                            # checkall the other dims if all = 1
                            if all(dim.dim_value == 1 for dim in dims[:-1]):
                                channel_info[layer_name]["parallel_window"] = 0
                            else:
                                channel_info[layer_name]["parallel_window"] = 1
                #else:
                    #channel_info[layer_name]["out_channels"] = onnx_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
                    #channel_info[layer_name]["parallel_window"] = 0
        
        # check if in_channels and out_channels are not set, set them to the graph input and output
        if "in_channels" not in channel_info[layer_name]:
            if onnx_model.graph.input:
                channel_info[layer_name]["in_channels"] = onnx_model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        if "out_channels" not in channel_info[layer_name]:
            if onnx_model.graph.output:
                channel_info[layer_name]["out_channels"] = onnx_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value

    return channel_info

def get_onnx_model(model_path):
    model = ModelWrapper(model_path)
    #model = model.transform(GiveUniqueNodeNames())
    return model

def cycle_analysis(model, plot = False):
    """
    Perform cycle analysis on the model.
    """
    cycles_dict = model.analysis(exp_cycles_per_layer)
    cycles_dict
    if plot:
        fig = plt.figure(figsize=(10, 5))
        plt.bar(cycles_dict.keys(), cycles_dict.values(), color='blue', width=0.3)
        plt.xlabel("Network layers")
        plt.ylabel("Number of clock cycles")
        plt.title("Clock cycles per layer PE=SIMD=1")
        plt.show()
        print(spliter)
        print("Cycle analysis for the model")
        print(spliter)
        print(cycles_dict)
    return cycles_dict

def resource_analysis(model, fpgapart = pynq_part_map["XCvu9P"], plot = False):
    res_dict = model.analysis(partial(res_estimation, fpgapart=fpgapart))
    transformed_dict = {key: value['LUT'] for key, value in res_dict.items()}
    LUTs = [res_dict[key]["LUT"] for key in res_dict.keys()] 
    if plot:
        fig = plt.figure(figsize = (10, 5))
        plt.bar(res_dict.keys(), LUTs, color ='green', width = 0.3)
        plt.xlabel("Network layers")
        plt.ylabel("Number of LUTs")
        plt.title("No. of LUTs per layer PE=SIMD=1")
        plt.show() 
        print(spliter)
        print("Resource estimation for the model on the FPGA part: " + fpgapart)
        print(spliter)
        print(transformed_dict)
    return transformed_dict

def check_attributes(attr_dict):
    has_simd = 'SIMD' in attr_dict
    has_pe = 'PE' in attr_dict
    has_parallel_window = 'parallel_window' in attr_dict
    return has_simd, has_pe, has_parallel_window


def modify_mvau_parallelization(model, node_name, pe=1, simd=1):
    """
    Modify the MVAU parallelization parameters in the model.
    """
    node_inst = getCustomOp(model.get_node_from_name(node_name))
    # print("The parallelization parameters of %s were: " % node_name)
    node_attrs = node_inst.get_nodeattr_types()
    # print(node_attrs)
    has_simd, has_pe, has_parallel_window = check_attributes(node_attrs)
    if has_pe:
        #print("Original PE: " + str(node_inst.get_nodeattr("PE")))
        node_inst.set_nodeattr("PE", pe)
        #print("Modified PE: " + str(node_inst.get_nodeattr("PE")))
    if has_simd:
        #print("Original SIMD: " + str(node_inst.get_nodeattr("SIMD")))
        node_inst.set_nodeattr("SIMD", simd)
        #print("Modified SIMD: " + str(node_inst.get_nodeattr("SIMD")))

# def unfold_node(model, node_name, channel_info):
#     """
#     Modify the MVAU parallelization parameters in the model.
#     """
#     node_inst = getCustomOp(model.get_node_from_name(node_name))
#     print("The parallelization parameters of %s were: " % node_name)
#     node_attrs = node_inst.get_nodeattr_types()
#     has_simd, has_pe, has_parallel_window = check_attributes(node_attrs)

#     channels = channel_info[node_name]
#     if has_pe:
#         print("Original PE: " + str(node_inst.get_nodeattr("PE")))
#         node_inst.set_nodeattr("PE", channels["out_channels"])
#         print("Modified PE: " + str(channels["out_channels"]))
#     if has_simd:
#         print("Original SIMD: " + str(node_inst.get_nodeattr("SIMD")))
#         node_inst.set_nodeattr("SIMD", channels["in_channels"])
#         print("Modified SIMD: " + str(channels["in_channels"]))
#     if has_parallel_window:
#         node_inst.set_nodeattr("parallel_window", channels["parallel_window"])
def unfold_node(model, node_name, channel_info, unfold_mode="full"):
    """
    Modify the MVAU parallelization parameters in the model
    unfold_mode: 
        "full" - 完全展开 (设置PE和SIMD为最大值)
        "factor" - 按因子展开 (已弃用，保留兼容)
        int - 指定展开因子
    """
    node_inst = getCustomOp(model.get_node_from_name(node_name))
    print(f"Modifying {node_name} with mode: {unfold_mode}")
    
    node_attrs = node_inst.get_nodeattr_types()
    has_simd, has_pe, has_parallel_window = check_attributes(node_attrs)

    channels = channel_info[node_name]
    
    if has_pe:
        orig_pe = node_inst.get_nodeattr("PE")
        if unfold_mode == "full":
            new_pe = channels["out_channels"]
        elif isinstance(unfold_mode, int):
            new_pe = min(orig_pe * unfold_mode, channels["out_channels"])
        else:
            new_pe = orig_pe  # 默认不改变
            
        print(f"Original PE: {orig_pe}, Modified PE: {new_pe}")
        node_inst.set_nodeattr("PE", new_pe)
    
    if has_simd:
        orig_simd = node_inst.get_nodeattr("SIMD")
        if unfold_mode == "full":
            new_simd = channels["in_channels"]
        elif isinstance(unfold_mode, int):
            new_simd = min(orig_simd * unfold_mode, channels["in_channels"])
        else:
            new_simd = orig_simd  # 默认不改变
            
        print(f"Original SIMD: {orig_simd}, Modified SIMD: {new_simd}")
        node_inst.set_nodeattr("SIMD", new_simd)
    
    if has_parallel_window:
        node_inst.set_nodeattr("parallel_window", channels["parallel_window"])
    
    return new_pe, new_simd


def get_node_names(model):
    """
    Get the names of all nodes in the model.
    """
    node_names = [node.name for node in model.graph.node]
    return node_names

def find_bottle_neck(cycle_result):
    """
    Find the bottleneck in the cycle result and return their indices.
    Assumes cycle_result is an ordered dict or list where order corresponds to layer indices.
    """
    if isinstance(cycle_result, dict):
        items = list(cycle_result.items())
    elif isinstance(cycle_result, list):
        items = list(enumerate(cycle_result))
    else:
        raise ValueError("Input must be a dict or list")
    
    max_cycles = max(cycle for _, cycle in items)
    bottleneck_indices = [idx for idx, cycle in items if cycle == max_cycles]

    # only maintain the first bottleneck index
    if bottleneck_indices:
        bottleneck_indices = bottleneck_indices[0]
        #max_cycles = max_cycles[bottleneck_indices[0]]

    #print(f"Bottleneck indices: {bottleneck_indices} with {max_cycles} cycles")
    return bottleneck_indices, max_cycles



def find_total_resources(resource_result):
    """
    Find the total resources used in the model.
    """
    total_resources = sum(resource_result.values())
    print(f"Total resources used: {total_resources}")
    return total_resources

def check_sparsity(onnx_model):
    """
    Check the sparsity of the model.
    """
    total_params = 0
    pruned_params = 0
    for node in onnx_model.graph.node:
        if node.op_type == "QuantLinear" or node.op_type == "QuantConv2d":
            weight_tensor = onnx_model.get_initializer(node.input[1])
            total_params += weight_tensor.size
            pruned_params += (weight_tensor == 0).sum().item()
    
    sparsity = pruned_params / total_params if total_params > 0 else 0
    print(f"Sparsity: {sparsity:.2f} (Pruned: {pruned_params}, Total: {total_params})")
    return sparsity, total_params, pruned_params


def find_bottle_neck_excluding(cycle_dict, skip_indices):
    """Find bottleneck layer excluding skipped indices"""
    candidate_layers = []
    for layer, cycles in cycle_dict.items():
        try:
            layer_idx = int(layer.split('_')[-1])
            if layer_idx not in skip_indices:
                candidate_layers.append((layer, cycles, layer_idx))
        except (ValueError, IndexError):
            continue
    
    if not candidate_layers:
        return None, 0
    
    # Sort by cycle count descending
    candidate_layers.sort(key=lambda x: x[1], reverse=True)
    return candidate_layers[0][0], candidate_layers[0][1]

def find_bottle_neck_excluding(cycle_dict, skip_indices):
    """Find bottleneck layer excluding skipped indices"""
    candidate_layers = []
    for layer, cycles in cycle_dict.items():
        try:
            layer_idx = int(layer.split('_')[-1])
            if layer_idx not in skip_indices:
                candidate_layers.append((layer, cycles, layer_idx))
        except (ValueError, IndexError):
            continue
    
    if not candidate_layers:
        return None, 0
    
    # Sort by cycle count descending
    candidate_layers.sort(key=lambda x: x[1], reverse=True)
    return candidate_layers[0][0], candidate_layers[0][1]

# def solve_bottle_neck(model_path, sparsity_info, fpgapart="xcvu9p-flgb2104-2-i"):
#     # Load and prepare the model
#     model = get_onnx_model(model_path)
#     onnx_model = onnx.load(model_path)
#     channel_info = get_layer_channels(onnx_model)
    
#     # FPGA resource configuration
#     fpga_resources = {
#         "xcu50-fsvh2104-2L-e": 1728000,
#         "xcvu9p-flgb2104-2-i": 2586000
#     }
#     available_luts = fpga_resources.get(fpgapart, 2586000)  # Default to xcvu9p
    
#     # Initial model analysis
#     auto_cycle = cycle_analysis(model, plot=False)
#     auto_res = resource_analysis(model, fpgapart=fpgapart, plot=False)
    
#     # Convert list-based sparsity info to layer-keyed dictionary
#     sparsity_dict = {}
#     for i, sparsity in enumerate(sparsity_info):
#         layer_name = f'MVAU_hls_{i}'
#         sparsity_dict[layer_name] = sparsity
    
#     print(f"{spliter}\nSparsity Information (by layer):")
#     for layer, sparsity in sparsity_dict.items():
#         print(f"{layer}: {sparsity}%")
    
#     print(f"{spliter}\nAuto Cycle Result:\n{auto_cycle}")
#     print(f"{spliter}\nAuto Resource Result:\n{auto_res}")
    
#     # Create unfolded model for comparison
#     unfold_model = get_onnx_model(model_path)
#     node_names = get_node_names(unfold_model)
#     for node_name in node_names:
#         unfold_node(unfold_model, node_name, channel_info)
    
#     unfold_cycle = cycle_analysis(unfold_model, plot=False)
#     unfold_res = resource_analysis(unfold_model, fpgapart=fpgapart, plot=True)
    
#     print(f"{spliter}\nUnfold Cycle Result:\n{unfold_cycle}")
#     print(f"{spliter}\nUnfold Resource Result:\n{unfold_res}{spliter}")
    
#     # Step 1: Apply sparsity-aware unfolding decisions
#     num_layers = len(auto_res)
#     unfold_decision = [0] * num_layers
#     sparsity_decision = [0] * num_layers
#     scaled_unfold_res = {}
    
#     for i in range(num_layers):
#         layer_name = f'MVAU_hls_{i}'
        
#         # Calculate sparsity-scaled resources
#         sparsity_ratio = (100 - sparsity_dict[layer_name]) / 100.0
#         scaled_unfold_res[layer_name] = unfold_res[layer_name] * sparsity_ratio
        
#         # Decide whether to unfold based on resource savings
#         if scaled_unfold_res[layer_name] <= auto_res[layer_name]:
#             unfold_decision[i] = 1
#             sparsity_decision[i] = 1
    
#     print("Scaled Unfold Resources (by layer):")
#     for layer, res in scaled_unfold_res.items():
#         print(f"{layer}: {res:.2f} LUTs")
    
#     print(f"Auto Resources (by layer):")
#     for layer, res in auto_res.items():
#         print(f"{layer}: {res} LUTs")
    
#     print(f"Unfold Decision: {unfold_decision}")
#     print(f"Sparsity Decision: {sparsity_decision}")
    
#     # Create modified model with unfolding decisions applied
#     mod_model = get_onnx_model(model_path)
#     node_names = get_node_names(mod_model)
#     for i, node_name in enumerate(node_names):
#         if unfold_decision[i] == 1:
#             unfold_node(mod_model, node_name, channel_info)
#             print(f"Unfolded node: {node_name}")
    
#     # Analyze modified model
#     mod_cycle = cycle_analysis(mod_model, plot=False)
#     mod_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
    
#     print(f"{spliter}\nModified Cycle Result:\n{mod_cycle}")
#     print(f"{spliter}\nModified Resource Result:\n{mod_res}{spliter}")
    
#     # Step 2: Iterative bottleneck optimization
#     # Step 2: Iterative bottleneck optimization
#     optimized = True
#     iteration = 1
#     skip_layers = set()  # Track layers that can't be unfolded further
#     double_unfold = set()   # Track layers with double unfolding
    
#     print(f"{spliter}\nStarting iterative optimization with {len(node_names)} layers")
#     print(f"Initial skip layers: {skip_layers}")
    
#     while optimized:
#         print(f"\n{spliter}\nIteration {iteration}: Bottleneck Optimization")
#         print(f"Skipped layers: {skip_layers}")
#         print(f"Double unfolded layers: {double_unfold}")
        
#         # Find current bottleneck excluding skipped layers
#         bottleneck_layer, max_cycle = find_bottle_neck_excluding(mod_cycle, skip_layers)
        
#         # Exit condition: no more optimizable layers
#         if bottleneck_layer is None:
#             print("No optimizable layers left. Exiting optimization loop.")
#             optimized = False
#             break
        
#         bottleneck_index = int(bottleneck_layer.split('_')[-1])
#         print(f"Current bottleneck: {bottleneck_layer} with {max_cycle} cycles")
        
#         # DEBUG: Show all candidate layers
#         print("Candidate bottleneck layers:")
#         candidate_layers = []
#         for layer, cycles in mod_cycle.items():
#             try:
#                 layer_idx = int(layer.split('_')[-1])
#                 if layer_idx not in skip_layers:
#                     candidate_layers.append((layer, cycles, layer_idx))
#             except (ValueError, IndexError):
#                 continue
        
#         # Sort by cycle count descending
#         candidate_layers.sort(key=lambda x: x[1], reverse=True)
#         for layer, cycles, idx in candidate_layers:
#             status = ""
#             if idx in double_unfold:
#                 status = " (double unfolded)"
#             elif unfold_decision[idx] == 1:
#                 status = " (unfolded)"
#             print(f"  {layer}: {cycles} cycles{status}")
        
#         # Check if we can optimize this layer
#         if (bottleneck_index in skip_layers or 
#             bottleneck_index >= len(unfold_decision)):
#             print(f"Skipping layer {bottleneck_layer} (marked to skip)")
#             skip_layers.add(bottleneck_index)
#             continue
        
#         # Calculate resource impact of unfolding this layer
#         layer_name = f'MVAU_hls_{bottleneck_index}'
#         current_res = mod_res[layer_name]
        
#         # Check if we should do double unfold (if previously attempted)
#         unfold_factor = 1
#         if bottleneck_index in double_unfold:
#             # Already double unfolded - skip this layer
#             print(f"Layer {bottleneck_index} already double unfolded. Skipping.")
#             skip_layers.add(bottleneck_index)
#             continue
#         elif unfold_decision[bottleneck_index] == 1:
#             # Already unfolded once - try double unfold
#             unfold_factor = 2
#             print(f"Attempting double unfold on {layer_name}")
#         else:
#             print(f"Attempting unfold on {layer_name}")
        
#         # Temporary unfold decision for this layer
#         temp_unfold_decision = unfold_decision.copy()
#         temp_unfold_decision[bottleneck_index] = unfold_factor
        
#         # Create temporary model with new unfold decision
#         temp_model = get_onnx_model(model_path)
#         for i, node_name in enumerate(node_names):
#             if temp_unfold_decision[i] > 0:
#                 # Handle unfolding based on factor
#                 if temp_unfold_decision[i] == 2:
#                     # First unfold
#                     unfold_node(temp_model, node_name, channel_info)
#                     # Second unfold (on already unfolded node)
#                     unfold_node(temp_model, node_name, channel_info)
#                 else:
#                     unfold_node(temp_model, node_name, channel_info)
        
#         # Analyze temporary model
#         temp_res = resource_analysis(temp_model, fpgapart=fpgapart, plot=False)
        
#         # Apply sparsity scaling to unfolded resources
#         total_resources = 0
#         for layer, res in temp_res.items():
#             try:
#                 idx = int(layer.split('_')[-1])
#                 if temp_unfold_decision[idx] > 0 and sparsity_decision[idx] == 1:
#                     sparsity_ratio = (100 - sparsity_dict[layer]) / 100.0
#                     total_resources += res * sparsity_ratio
#                 else:
#                     total_resources += res
#             except (ValueError, IndexError):
#                 total_resources += res
        
#         print(f"Projected total LUTs: {total_resources:.2f}/{available_luts}")
        
#         # Check resource constraints
#         if total_resources <= available_luts:
#             print(f"Resource constraint satisfied. Applying {'double ' if unfold_factor > 1 else ''}unfold to {layer_name}")
            
#             # Update model and decisions
#             mod_model = temp_model
#             unfold_decision = temp_unfold_decision
#             if unfold_factor > 1:
#                 double_unfold.add(bottleneck_index)
#                 print(f"Marked layer {bottleneck_index} as double unfolded")
            
#             # Reanalyze model
#             mod_cycle = cycle_analysis(mod_model, plot=False)
#             mod_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
            
#             print(f"New bottleneck cycle: {max(mod_cycle.values())}")
#             print(f"New resource total: {sum(mod_res.values())}")
#         else:
#             print(f"Resource constraint violated ({total_resources:.2f} > {available_luts})")
            
#             if unfold_factor == 1:
#                 # First violation - try double unfold
#                 print("Attempting double unfold to reduce resource usage")
#                 double_unfold.add(bottleneck_index)
#             else:
#                 # Still violates after double unfold - skip this layer
#                 print("Double unfold still violates constraints. Skipping layer.")
#                 skip_layers.add(bottleneck_index)
        
#         # Update iteration counter
#         iteration += 1
        
#         # Safety break to prevent infinite loops
#         if iteration > len(node_names) * 3:  # Max 3 attempts per layer
#             print("Reached maximum iteration limit. Exiting optimization loop.")
#             optimized = False
        
#         # Check exit condition: no more optimizable layers
#         if len(skip_layers) >= len(node_names):
#             print("All layers processed. Exiting optimization loop.")
#             optimized = False
    
#     # Final analysis
#     final_cycle = cycle_analysis(mod_model, plot=True)
#     final_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
#     final_bottleneck, max_cycle = find_bottle_neck(final_cycle)
    
#     print(f"{spliter}\nFINAL RESULTS")
#     print(f"Bottleneck layer: {final_bottleneck} with {max_cycle} cycles")
#     print(f"Total LUTs used: {sum(final_res.values())}/{available_luts}")
    
#     # Export configurations
#     folding_config = []
#     sparsity_config = []
    
#     for i in range(len(unfold_decision)):
#         layer_name = f'MVAU_hls_{i}'
#         folding_level = unfold_decision[i]
#         sparsity_level = sparsity_decision[i] if unfold_decision[i] > 0 else 0
        
#         folding_config.append({
#             "layer": layer_name,
#             "folding_level": folding_level,
#             "double_unfold": i in double_unfold
#         })
        
#         sparsity_config.append({
#             "layer": layer_name,
#             "sparsity_used": sparsity_level,
#             "sparsity_ratio": sparsity_dict[layer_name]
#         })
    
#     print(f"{spliter}\nFOLDING CONFIGURATION:")
#     for config in folding_config:
#         print(f"{config['layer']}: Level {config['folding_level']} {'(Double)' if config['double_unfold'] else ''}")
    
#     print(f"{spliter}\nSPARSITY CONFIGURATION:")
#     for config in sparsity_config:
#         print(f"{config['layer']}: {'Enabled' if config['sparsity_used'] else 'Disabled'} "
#               f"(Sparsity: {config['sparsity_ratio']}%)")
    
#     return {
#         "final_model": mod_model,
#         "folding_config": folding_config,
#         "sparsity_config": sparsity_config,
#         "bottleneck": final_bottleneck,
#         "max_cycle": max_cycle,
#         "total_resources": sum(final_res.values()),
#         "available_resources": available_luts,
#         "resource_utilization": sum(final_res.values()) / available_luts
#     }
def solve_bottle_neck(model_path, sparsity_info, fpgapart="xcvu9p-flgb2104-2-i"):
    # Load and prepare the model
    model = get_onnx_model(model_path)
    onnx_model = onnx.load(model_path)
    channel_info = get_layer_channels(onnx_model)
    
    # FPGA resource configuration
    fpga_resources = {
        "xcu50-fsvh2104-2L-e": 1728000,
        "xcvu9p-flgb2104-2-i": 2586000
    }
    available_luts = fpga_resources.get(fpgapart, 2586000)  # Default to xcvu9p
    
    # Initial model analysis
    auto_cycle = cycle_analysis(model, plot=False)
    auto_res = resource_analysis(model, fpgapart=fpgapart, plot=False)
    
    # Convert list-based sparsity info to layer-keyed dictionary
    sparsity_dict = {}
    for i, sparsity in enumerate(sparsity_info):
        layer_name = f'MVAU_hls_{i}'
        sparsity_dict[layer_name] = sparsity
    
    print(f"{spliter}\nSparsity Information (by layer):")
    for layer, sparsity in sparsity_dict.items():
        print(f"{layer}: {sparsity}%")
    
    print(f"{spliter}\nAuto Cycle Result:\n{auto_cycle}")
    print(f"{spliter}\nAuto Resource Result:\n{auto_res}")
    
    # Create unfolded model for comparison
    unfold_model = get_onnx_model(model_path)
    node_names = get_node_names(unfold_model)
    for node_name in node_names:
        unfold_node(unfold_model, node_name, channel_info, unfold_mode='full')  # Fully unfold
    
    unfold_cycle = cycle_analysis(unfold_model, plot=False)
    unfold_res = resource_analysis(unfold_model, fpgapart=fpgapart, plot=True)
    
    print(f"{spliter}\nUnfold Cycle Result:\n{unfold_cycle}")
    print(f"{spliter}\nUnfold Resource Result:\n{unfold_res}{spliter}")
    
    # Step 1: Apply sparsity-aware unfolding decisions
    num_layers = len(auto_res)
    unfold_decision = [0] * num_layers
    sparsity_decision = [0] * num_layers
    scaled_unfold_res = {}
    
    for i in range(num_layers):
        layer_name = f'MVAU_hls_{i}'
        
        # Calculate sparsity-scaled resources
        sparsity_ratio = (100 - sparsity_dict[layer_name]) / 100.0
        scaled_unfold_res[layer_name] = unfold_res[layer_name] * sparsity_ratio
        
        # Decide whether to unfold based on resource savings
        if scaled_unfold_res[layer_name] <= auto_res[layer_name]:
            unfold_decision[i] = 1
            sparsity_decision[i] = 1
    
    print("Scaled Unfold Resources (by layer):")
    for layer, res in scaled_unfold_res.items():
        print(f"{layer}: {res:.2f} LUTs")
    
    print(f"Auto Resources (by layer):")
    for layer, res in auto_res.items():
        print(f"{layer}: {res} LUTs")
    
    print(f"Unfold Decision: {unfold_decision}")
    print(f"Sparsity Decision: {sparsity_decision}")
    
    # Create modified model with unfolding decisions applied
    mod_model = get_onnx_model(model_path)
    node_names = get_node_names(mod_model)
    for i, node_name in enumerate(node_names):
        if unfold_decision[i] == 1:
            unfold_node(mod_model, node_name, channel_info, unfold_mode='full')
            print(f"Unfolded node: {node_name}")
    
    # Analyze modified model
    mod_cycle = cycle_analysis(mod_model, plot=False)
    mod_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
    
    print(f"{spliter}\nModified Cycle Result:\n{mod_cycle}")
    print(f"{spliter}\nModified Resource Result:\n{mod_res}{spliter}")
    
    # Step 2: 迭代瓶颈优化
    optimized = True
    iteration = 1
    skip_layers = set()  # 跟踪无法进一步展开的层
    double_unfold = set()   # 跟踪有额外展开的层

    print(f"{spliter}\nStarting iterative optimization with {len(node_names)} layers")
    print(f"Initial skip layers: {skip_layers}")

    while optimized:
        print(f"\n{spliter}\nIteration {iteration}: Bottleneck Optimization")
        print(f"Skipped layers: {skip_layers}")
        print(f"Additional unfolded layers: {double_unfold}")
        
        # 查找当前瓶颈（排除已跳过的层）
        bottleneck_layer, max_cycle = find_bottle_neck_excluding(mod_cycle, skip_layers)
        
        # 退出条件：没有可优化的层
        if bottleneck_layer is None:
            print("No optimizable layers left. Exiting optimization loop.")
            optimized = False
            break
        
        bottleneck_index = int(bottleneck_layer.split('_')[-1])
        layer_name = f'MVAU_hls_{bottleneck_index}'
        print(f"Current bottleneck: {layer_name} with {max_cycle} cycles")
        
        # 检查是否可以优化该层
        if (bottleneck_index in skip_layers or 
            bottleneck_index >= len(unfold_decision)):
            print(f"Skipping layer {layer_name} (marked to skip)")
            skip_layers.add(bottleneck_index)
            continue
        
        # 获取当前层属性
        current_pe = get_layer_pe(mod_model, layer_name)
        max_pe = channel_info[layer_name]["out_channels"]
        current_simd = get_layer_simd(mod_model, layer_name)
        max_simd = channel_info[layer_name]["in_channels"]
        
        # 检查是否已达到最大展开
        if current_pe >= max_pe and current_simd >= max_simd:
            print(f"Layer {layer_name} already fully unfolded. Skipping.")
            skip_layers.add(bottleneck_index)
            continue
        
        # 确定展开因子 - 对于已展开的层加倍
        unfold_factor = 2 if bottleneck_index in double_unfold else 1
        print(f"Attempting unfold factor {unfold_factor} on {layer_name}")
        
        # 临时展开决策
        temp_unfold_decision = unfold_decision.copy()
        
        # 如果尚未展开，则设置为完全展开
        if temp_unfold_decision[bottleneck_index] == 0:
            temp_unfold_decision[bottleneck_index] = 1  # 表示完全展开
        else:
            temp_unfold_decision[bottleneck_index] += unfold_factor
        
        # 创建临时模型
        temp_model = get_onnx_model(model_path)
        
        # 应用所有展开决策到临时模型
        for i, node_name in enumerate(node_names):
            if temp_unfold_decision[i] == 1:
                unfold_node(temp_model, node_name, channel_info, unfold_mode="full")
            elif temp_unfold_decision[i] > 1:
                unfold_node(temp_model, node_name, channel_info, unfold_mode=temp_unfold_decision[i])
        
        # 分析临时模型资源
        temp_res = resource_analysis(temp_model, fpgapart=fpgapart, plot=False)
        
        # 应用稀疏缩放
        total_resources = 0
        for layer, res in temp_res.items():
            try:
                idx = int(layer.split('_')[-1])
                if temp_unfold_decision[idx] > 0 and sparsity_decision[idx] == 1:
                    sparsity_ratio = (100 - sparsity_dict[layer]) / 100.0
                    total_resources += res * sparsity_ratio
                else:
                    total_resources += res
            except (ValueError, IndexError):
                total_resources += res
        
        print(f"Projected total LUTs: {total_resources:.2f}/{available_luts}")
        
        # 检查资源约束
        if total_resources <= available_luts:
            print(f"Resource constraint satisfied. Applying changes to {layer_name}")
            
            # 更新模型和决策
            mod_model = temp_model
            unfold_decision = temp_unfold_decision
            double_unfold.add(bottleneck_index)
            
            # 重新分析模型
            mod_cycle = cycle_analysis(mod_model, plot=False)
            mod_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
            
            print(f"New bottleneck cycle: {max(mod_cycle.values())}")
            print(f"New resource total: {sum(mod_res.values())}")
        else:
            print(f"Resource constraint violated ({total_resources:.2f} > {available_luts})")
            
            if unfold_factor == 1:
                # 首次违反 - 尝试较小展开因子
                print("Attempting smaller unfold factor to reduce resource usage")
                skip_layers.add(bottleneck_index)
            else:
                # 仍然违反 - 跳过该层
                print("Unfolding still violates constraints. Skipping layer.")
                skip_layers.add(bottleneck_index)
        
        # 更新迭代计数器
        iteration += 1
        
        # 安全中断防止无限循环
        if iteration > len(node_names) * 3:  # 每层最多尝试3次
            print("Reached maximum iteration limit. Exiting optimization loop.")
            optimized = False
        
        # 检查退出条件：没有更多可优化层
        if len(skip_layers) >= len(node_names):
            print("All layers processed. Exiting optimization loop.")
            optimized = False
    
    # Final analysis
    final_cycle = cycle_analysis(mod_model, plot=True)
    final_res = resource_analysis(mod_model, fpgapart=fpgapart, plot=True)
    final_bottleneck, max_cycle = find_bottle_neck(final_cycle)
    
    # Calculate final resource usage with sparsity scaling
    total_luts = 0
    for layer, res in final_res.items():
        try:
            idx = int(layer.split('_')[-1])
            if unfold_decision[idx] > 0 and sparsity_decision[idx] == 1:
                sparsity_ratio = (100 - sparsity_dict[layer]) / 100.0
                total_luts += res * sparsity_ratio
            else:
                total_luts += res
        except (ValueError, IndexError):
            total_luts += res

    print(f"{spliter}\nFINAL RESULTS")
    print(f"Bottleneck layer: {final_bottleneck} with {max_cycle} cycles")
    print(f"Total LUTs used (with sparsity scaling): {total_luts:.2f}/{available_luts}")
    print(f"Resource utilization: {total_luts/available_luts:.2%}")
    
    # Export configurations
    folding_config = []
    sparsity_config = []
    
    for i in range(len(unfold_decision)):
        layer_name = f'MVAU_hls_{i}'
        folding_level = unfold_decision[i]
        
        # Determine fold type string
        if folding_level == 0:
            fold_type = "FOLDED"
        elif folding_level == 1:
            fold_type = "UNFOLD(1)"
        else:
            fold_type = f"UNFOLD({folding_level})"
        
        folding_config.append({
            "layer": layer_name,
            "folding_level": folding_level,
            "fold_type": fold_type,
            "additional_unfold": i in double_unfold
        })
        
        sparsity_config.append({
            "layer": layer_name,
            "sparsity_used": sparsity_decision[i],
            "sparsity_ratio": sparsity_dict[layer_name]
        })

    print(f"{spliter}\nFOLDING CONFIGURATION:")
    for config in folding_config:
        print(f"{config['layer']}: {config['fold_type']} {'(Additional)' if config['additional_unfold'] else ''}")

    print(f"{spliter}\nSPARSITY CONFIGURATION:")
    for config in sparsity_config:
        print(f"{config['layer']}: {'Enabled' if config['sparsity_used'] else 'Disabled'} "
              f"(Sparsity: {config['sparsity_ratio']}%)")
    
    # export the final model to ONNX format
    # final_bottleneck = final_bottleneck if final_bottleneck else "None"
    # onnx.save_model(mod_model, "final_model.onnx")
    # print(f"Final model saved as 'final_model.onnx'")   
    mod_model.save("final_model2.onnx")

    return {
        "final_model": mod_model,
        "folding_config": folding_config,
        "sparsity_config": sparsity_config,
        "bottleneck": final_bottleneck,
        "max_cycle": max_cycle,
        "total_resources": total_luts,
        "available_resources": available_luts,
        "resource_utilization": total_luts / available_luts
    }



# Helper functions for layer attribute retrieval
def get_layer_pe(model, layer_name):
    node_inst = getCustomOp(model.get_node_from_name(layer_name))
    return node_inst.get_nodeattr("PE")

def get_layer_simd(model, layer_name):
    node_inst = getCustomOp(model.get_node_from_name(layer_name))
    return node_inst.get_nodeattr("SIMD")


if __name__ == "__main__":
    # load the model
    model_path = "/home/changhong/prj/finn/script/EF_US/casestudy1/step_generate_estimate_reports.onnx"
    model = get_onnx_model(model_path)
    onnx_model = onnx.load(model_path)
    channel_info = get_layer_channels(onnx_model)
    #print("Channel information for the model:")
    #print(channel_info)
    # check latency and resources bottle neck
    print(spliter)
    print('Analyzing auto fold model')
    print(spliter)
    auto_cycle_result = cycle_analysis(model)
    auto_res_result = resource_analysis(model, fpgapart = pynq_part_map["U50"])
    print("\n\n")
    # fully-fold the model
    node_names = get_node_names(model)
    #print("Node names in the model:")
    #print(node_names)
    for node_name in node_names:
        # print("Modifying parallelization for node: " + node_name)
        modify_mvau_parallelization(model, node_name, pe=1, simd=1)
        # re-analyze the model after modification
    print(spliter)
    print('Analyzing auto fully folded model')
    print(spliter)
    fold_cycle_result = cycle_analysis(model, plot=True)
    fold_res_result = resource_analysis(model, fpgapart=pynq_part_map["U50"], plot=True)
    print("\n\n")
    # fully unfold the model
    for node_name in node_names:
        # print("Unfolding node: " + node_name)
        unfold_node(model, node_name, channel_info)
        # re-analyze the model after unfolding
    print(spliter)
    print('Analyzing auto fully unfolded model')
    print(spliter)
    unfold_cycle_result = cycle_analysis(model, plot=True)
    unfold_res_result = resource_analysis(model, fpgapart=pynq_part_map["U50"], plot=True)
    print("\n\n")
    # find the bottle neck and adjust folding config and saprsity decision