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

def resource_analysis(model, fpgapart = pynq_part_map["U50"], plot = False):
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

def unfold_node(model, node_name, channel_info):
    """
    Modify the MVAU parallelization parameters in the model.
    """
    node_inst = getCustomOp(model.get_node_from_name(node_name))
    print("The parallelization parameters of %s were: " % node_name)
    node_attrs = node_inst.get_nodeattr_types()
    has_simd, has_pe, has_parallel_window = check_attributes(node_attrs)

    channels = channel_info[node_name]
    if has_pe:
        print("Original PE: " + str(node_inst.get_nodeattr("PE")))
        node_inst.set_nodeattr("PE", channels["out_channels"])
        print("Modified PE: " + str(channels["out_channels"]))
    if has_simd:
        print("Original SIMD: " + str(node_inst.get_nodeattr("SIMD")))
        node_inst.set_nodeattr("SIMD", channels["in_channels"])
        print("Modified SIMD: " + str(channels["in_channels"]))
    if has_parallel_window:
        #print("Original parallel_window: " + str(node_inst.get_nodeattr("parallel_window")))
        node_inst.set_nodeattr("parallel_window", channels["parallel_window"])
        #print("Modified parallel_window: " + str(channels["parallel_window"]))

def get_node_names(model):
    """
    Get the names of all nodes in the model.
    """
    node_names = [node.name for node in model.graph.node]
    return node_names

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