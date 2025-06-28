import json
import onnx
import numpy as np
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
    
    return channel_info

def modify_json_config(json_data, channel_info):
    """修改JSON配置中的层参数"""
    # 需要修改的层类型前缀
    # target_prefixes = ("MVAU_hls", "MVAU_rtl")
    
    for layer_name, layer_config in json_data.items():

        base_name = layer_name
        if base_name in channel_info:
            channels = channel_info[base_name]
            
            # 更新参数
            if "in_channels" in channels:
                if "SIMD" in layer_config:
                    layer_config["SIMD"] = channels["in_channels"]
            if "out_channels" in channels:
                if "PE" in layer_config:
                    layer_config["PE"] = channels["out_channels"]
            if "parallel_window" in layer_config:
                # let parallel_windows = 1
                layer_config["parallel_window"] = channel_info[layer_name]["parallel_window"]

            # 修改内存模式
            if "mem_mode" in layer_config:
                layer_config["mem_mode"] = "internal_embedded"
    
    return json_data

def main():

    folding_json_file = './casestudy1/estimates_output/test/auto_folding_config.json'
    onnx_model_file = './casestudy1/estimates_output/test/intermediate_models/step_apply_folding_config.onnx'


    with open(folding_json_file) as f:
        json_data = json.load(f)
    onnx_model = onnx.load(onnx_model_file)
    channel_info = get_layer_channels(onnx_model)
    modified_json = modify_json_config(json_data, channel_info)
    
    with open("foldedtest.json", "w") as f:
        json.dump(modified_json, f, indent=2)
    
    print("unfolded.json has been created with updated parameters.")

def auto_unfold_json(folding_json_file, onnx_model_file, target_json_file="unfolded.json"):
    """自动展开JSON配置文件"""
    with open(folding_json_file) as f:
        json_data = json.load(f)
    onnx_model = onnx.load(onnx_model_file)
    channel_info = get_layer_channels(onnx_model)
    modified_json = modify_json_config(json_data, channel_info)
    
    with open(target_json_file, "w") as f:
        json.dump(modified_json, f, indent=2)
    
    print(f"{target_json_file} has been created with updated parameters.")


if __name__ == "__main__":
    main()
