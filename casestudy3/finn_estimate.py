import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil
import os

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
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import os
import configparser
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import numpy as np


from models import get_model
from dataset import get_dataloaders
# from train import train_try


# ----- USER PARAS ------ #

def estimate_ip(model_name, model, weight, activation, try_name= "model_generation_test", folding_config_file = "auto"):
    # model_name = "2c3f_relu"
    # model_weight = "./model/final_2c3f_relu_w4_a4_pruned.pth"
    # epochs = 50
    # model_name = '2c3f_relu'
    # model_name = '2c3f'
    # 2c3f relu is ok for generation, but fully unfold not tested yet
    # weight = 4
    # activation = 4
    model_ready_name = model_name + '_w'+ str(weight) + '_a' + str(activation) + '_ready.onnx'
    # try_name = "/model_generation_test"
    # folding_config_file = "./folding_config/auto.json"

    # ----- END USER PARAS ------ #

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


    build_dir = "./build/" + try_name
    model_dir = "./model"
    data_dir = "./data"
    estimates_output_dir = f"./estimates_output/{model_name}_{weight}_{activation}_{try_name}"
    rtlsim_output_dir = f"./rtlsim_output/{try_name}"
    #tmp_path = "./tmp" + try_name

    # convert tmp_path to absolute path
    #tmp_path = os.path.abspath(tmp_path)
    build_dir = os.path.abspath(build_dir)
    model_dir = os.path.abspath(model_dir)
    data_dir = os.path.abspath(data_dir)
    estimates_output_dir = os.path.abspath(estimates_output_dir)
    rtlsim_output_dir = os.path.abspath(rtlsim_output_dir)



    # Create directories if they do not exist
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(estimates_output_dir, exist_ok=True)
    os.makedirs(rtlsim_output_dir, exist_ok=True)
    #os.environ["FINN_BUILD_DIR"] = tmp_path
    print(f"Data directory: {data_dir}")
    print(f"Build directory: {build_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Estimates output directory: {estimates_output_dir}")
    print(f"RTLSim output directory: {rtlsim_output_dir}")
    #print(f"Tmp directory: {tmp_path}")



    # test training
    #model = train_try(model_name=model_name, w=weight, a=activation, epochs=epochs, random_seed=1998)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Analyze model sparsity
    print("Analyzing model sparsity...")
    # model = get_model(model_name, weight, activation)
    # model.load_state_dict(torch.load(model_weight))
    model.to(device)



    # test model generation
    ready_model_filename = model_dir + "/" + model_ready_name

    input_shape = (1, 1, 32, 32)
    if model_name == 'unsw_fc':
        input_shape = (1, 593)

    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)
    #Move to CPU before export

    if model_name == 'unsw_fc':
        # For unsw_fc, we need to modify the input shape to match the model
            # add input quantization layer to avoid first layer weight cant be implemented
        pre_layer = qnn.QuantIdentity(bit_width=activation)
        model = torch.nn.Sequential(pre_layer, model)

    model.cpu()
    # Export to ONNX
    export_qonnx(
        model, export_path=ready_model_filename, input_t=input_t
    )
    # clean-up
    qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)
    print("Ready Model saved to %s" % ready_model_filename)

    if os.path.exists(estimates_output_dir):
        shutil.rmtree(estimates_output_dir)
        print("Previous run results deleted!")

    if os.path.exists(rtlsim_output_dir):
        shutil.rmtree(rtlsim_output_dir)
        print("Previous run results deleted!")


    cfg_estimates = build.DataflowBuildConfig(
        output_dir          = estimates_output_dir,
        target_fps          = 100000000,
        mvau_wwidth_max     = 10000, # test
        synth_clk_period_ns = 1.0,
        fpga_part           = pynq_part_map["XCvu9P"],
        steps               = build_cfg.estimate_only_dataflow_steps,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        ]
    )

    build.build_dataflow_cfg(ready_model_filename, cfg_estimates)

if __name__ == "__main__":
    # model_name = 'tfc'
    # weight = 1 
    # activation = 1
    # epochs = 10
    # model = train_try(model_name=model_name, w=weight, a=activation, epochs=epochs, random_seed=1998)
    # model = get_model(model_name, weight, activation)
    # model.load_state_dict(torch.load(f"./model/best_{model_name}_w{weight}_a{activation}_{epochs}.pth"))
    # #model.load_state_dict(torch.load(f"./model/final_{model_name}_w{weight}_a{activation}_l1_pruned.pth"))
    # #model.to(torch.device("cpu"))  # Ensure
    # estimate_ip(model_name=model_name, model = model, weight=1, activation=1, try_name="/test")
    model_name = 'unsw_fc'
    weight = 2 
    activation = 2
    epochs = 10
 
    model = get_model(model_name, weight, activation)
    model.load_state_dict(torch.load(f"./model/final_{model_name}_w{weight}_a{activation}_l1_pruned.pth"))
    model.to(torch.device("cpu"))  # Ensure

    estimate_ip(model_name=model_name, model = model, weight=weight, activation=activation, try_name="test5")
