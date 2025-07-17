from models import get_model
import torch
import numpy as np
import torch.nn as nn
import brevitas.nn as qnn

# 1. 创建新模型（包含 QuantIdentity）
new_model = get_model("unsw_fc", w=2, a=2)

# 2. 加载旧权重
old_weights = torch.load("pretrained/unsw_fc_w2_a2_pretrained copy.pth")

# 3. 创建新权重字典
new_state_dict = new_model.state_dict()

# 4. 迁移旧权重（考虑新增的输入量化层）
for name, param in old_weights.items():
    parts = name.split('.')
    # 由于新增了输入量化层，所有索引需要+1
    new_index = str(int(parts[0]) + 1)
    new_name = new_index + '.' + '.'.join(parts[1:])
    
    # 只迁移权重和偏置，不迁移量化参数
    if new_name in new_state_dict and "act_quant" not in name and "weight_quant" not in name:
        new_state_dict[new_name] = param

# 5. 加载修改后的权重到新模型
load_result = new_model.load_state_dict(new_state_dict, strict=False)
print("权重加载结果:")
print(f"缺失的键: {load_result.missing_keys}")
print(f"意外的键: {load_result.unexpected_keys}")

# 6. 手动初始化量化参数
quant_keys = [
    '0.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
    '4.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
    '8.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
    '12.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value'
]

for key in quant_keys:
    if key in new_model.state_dict():
        # 设置合理的初始缩放值
        new_model.state_dict()[key].data.fill_(1.0)
        print(f"已初始化量化参数: {key}")

# 7. 验证前向传播 - 使用至少2个样本避免BatchNorm问题
input_size = 593
test_input = torch.randn(2, input_size)  # 使用2个样本
new_model.eval()
with torch.no_grad():
    output = new_model(test_input)
    print("\n前向传播测试成功!")
    print(f"输出形状: {output.shape}")
    print(f"输出值示例: {output[0].item()}")

# 8. 保存完整的模型状态（包含量化参数）
torch.save(new_model.state_dict(), "pretrained/unsw_fc_w2_a2_pretrained.pth")
print("\n新权重已保存至: pretrained/unsw_fc_w2_a2_pretrained.pth")

# 9. 验证保存的权重（使用非严格模式）
print("\n验证保存的权重...")
test_model = get_model("unsw_fc", w=2, a=2)
test_model.eval()  # 确保在评估模式
saved_weights = torch.load("pretrained/unsw_fc_w2_a2_pretrained.pth")

# 检查保存的权重是否包含量化参数
quant_keys_in_saved = [key for key in quant_keys if key in saved_weights]
print(f"保存的权重中包含 {len(quant_keys_in_saved)}/{len(quant_keys)} 个量化参数")

try:
    # 使用非严格模式加载
    load_result_test = test_model.load_state_dict(saved_weights, strict=False)
    print("权重加载验证成功 (非严格模式)!")
    print(f"缺失的键: {load_result_test.missing_keys}")
    print(f"意外的键: {load_result_test.unexpected_keys}")
    
    # 前向传播验证 - 使用至少2个样本
    with torch.no_grad():
        test_output = test_model(test_input)
        print(f"验证输出值: {test_output[0].item()}")
        
        # 检查输出是否一致
        if torch.allclose(output, test_output, atol=1e-5):
            print("输出一致性验证通过!")
        else:
            print("警告: 两次输出不一致!")
            print(f"原始输出: {output[0].item()}, 验证输出: {test_output[0].item()}")
            
except Exception as e:
    print(f"权重加载失败: {str(e)}")
    
    # 详细诊断
    model_keys = set(test_model.state_dict().keys())
    saved_keys = set(saved_weights.keys())
    
    print("\n详细诊断:")
    print(f"模型需要的键: {len(model_keys)}")
    print(f"保存的权重包含的键: {len(saved_keys)}")
    print(f"缺失的键: {model_keys - saved_keys}")
    print(f"多余的键: {saved_keys - model_keys}")

# 10. 备选方案：保存模型权重和结构信息
print("\n保存模型权重和结构信息...")
model_info = {
    'state_dict': new_model.state_dict(),
    'model_config': {
        'w': 2,
        'a': 2,
        'model_type': 'unsw_fc'
    }
}
torch.save(model_info, "pretrained/unsw_fc_w2_a2_model_info.pth")
print("模型信息已保存至: pretrained/unsw_fc_w2_a2_model_info.pth")

# 测试加载模型信息
try:
    loaded_info = torch.load("pretrained/unsw_fc_w2_a2_model_info.pth")
    loaded_model = get_model(**loaded_info['model_config'])
    loaded_model.load_state_dict(loaded_info['state_dict'])
    loaded_model.eval()
    
    with torch.no_grad():
        loaded_output = loaded_model(test_input)
        print(f"加载模型输出值: {loaded_output[0].item()}")
        if torch.allclose(output, loaded_output, atol=1e-5):
            print("模型信息加载验证通过!")
except Exception as e:
    print(f"模型信息加载失败: {str(e)}")
