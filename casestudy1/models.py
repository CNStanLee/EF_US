import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d, AvgPool2d
from torch.nn import Module
from torch.nn import ModuleList
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas_examples.bnn_pynq.models.common import CommonActQuant
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant
from brevitas_examples.bnn_pynq.models.tensor_norm import TensorNorm
import os
from brevitas.core.scaling import ScalingImplType
from brevitas.core.quant import QuantType
import brevitas.nn as qnn
import ast
from functools import reduce
from operator import mul
from torch.nn import Dropout

DROPOUT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Current working directory: {os.getcwd()}")

import configparser

# CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
# INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
# LAST_FC_IN_FEATURES = 512
# LAST_FC_PER_OUT_CH_SCALING = False
# POOL_SIZE = 2
# KERNEL_SIZE = 3

# Model Name
model_name = '2c3f1w1a_mnist_new'
# LeNet-5
CNV_OUT_CH_POOL = [(6, True), (16, True), (120, False)]  
INTERMEDIATE_FC_FEATURES = [(120, 84)]  
LAST_FC_IN_FEATURES = 84 
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2  
KERNEL_SIZE = 5  
# Quant Configuration
config = configparser.ConfigParser()
config['MODEL'] = {
    'NUM_CLASSES': '10',
    'IN_CHANNELS': '1',
    'DTASET': 'MNIST',
}
config['QUANT'] = {
    'WEIGHT_BIT_WIDTH': '1',
    'ACT_BIT_WIDTH': '1',
    'IN_BIT_WIDTH': '8',
}

DROPOUT = 0.2

class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2)) # avg pool not supported in finn

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))

        self.linear_features.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x 
    
class CNV_RELU(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV_RELU, self).__init__()

        #self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            if act_bit_width == 1:
                self.conv_features.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            else:
                self.conv_features.append(
                    # qnn.QuantReLU(#quant_type=QuantType.INT, 
                    #                 bit_width=act_bit_width, # for relu, only support >2
                    #                 min_val=- 1.0,
                    #                 #max_val= 1- 1/128.0,
                    #                 max_val=1.0 - 2.0 ** (-7),
                    #                 restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                    #                 scaling_impl_type=ScalingImplType.CONST ))
                      qnn.QuantReLU(
                        bit_width=act_bit_width, return_quant_tensor=True))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2)) # avg pool not supported in finn

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            if act_bit_width == 1:
                self.linear_features.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            else:
                self.linear_features.append(
                    # qnn.QuantReLU(#quant_type=QuantType.INT, 
                    #                 bit_width=act_bit_width, # for relu, only support >2
                    #                 min_val=- 1.0,
                    #                 #max_val= 1- 1/128.0,
                    #                 max_val=1.0 - 2.0 ** (-7),
                    #                 restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                    #                 scaling_impl_type=ScalingImplType.CONST ))
                      qnn.QuantReLU(
                        bit_width=act_bit_width, return_quant_tensor=True))

        self.linear_features.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
            # if isinstance(mod, qnn.QuantReLU):
            #     mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)
            # if isinstance(mod, qnn.QuantReLU):
            #     mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x 





class FC(Module):

    def __init__(
        self,
        num_classes,
        weight_bit_width,
        act_bit_width,
        in_bit_width,
        in_channels,
        out_features,
        in_features=(32, 32)):
        super(FC, self).__init__()

        self.features = ModuleList()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width))
        self.features.append(Dropout(p=DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in out_features:
            self.features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features, eps=1e-4))
          
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            self.features.append(Dropout(p=DROPOUT))
        self.features.append(
            QuantLinear(
                in_features=in_features,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant))
        self.features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x
    
def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    net = CNV(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        num_classes=num_classes,
        in_ch=in_channels)
    return net

def fc(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    out_features = ast.literal_eval(cfg.get('MODEL', 'OUT_FEATURES'))
    net = FC(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        in_channels=in_channels,
        out_features=out_features,
        num_classes=num_classes)
    return net


def get_model(model_name, w, a):
    if model_name == '2c3f':
        net = CNV(
        weight_bit_width=w,
        act_bit_width=a,
        in_bit_width=8,
        num_classes=10,
        in_ch=1)
        return net
    elif model_name == '2c3f_relu':
        net = CNV_RELU(
        weight_bit_width=w,
        act_bit_width=a,
        in_bit_width=8,
        num_classes=10,
        in_ch=1)
        return net
    elif model_name == 'tfc':
        net = FC(
        weight_bit_width=w,
        act_bit_width=a,
        in_bit_width=8,
        in_channels=1,
        out_features=[64, 64, 64],
        num_classes=10)
        return net
    elif model_name == 'sfc':
        net = FC(
        weight_bit_width=w,
        act_bit_width=a,
        in_bit_width=8,
        in_channels=1,
        out_features=[256, 256, 256],
        num_classes=10)
        return net
    elif model_name == 'lfc':
        net = FC(
        weight_bit_width=w,
        act_bit_width=a,
        in_bit_width=8,
        in_channels=1,
        out_features=[1024, 1024, 1024],
        num_classes=10)
        return net
    else:
        raise ValueError(f"Model {model_name} is not supported.")
