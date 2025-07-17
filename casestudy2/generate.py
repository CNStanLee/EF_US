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
from brevitas.nn import QuantIdentity
from copy import deepcopy
from models import get_model
from dataset import get_dataloaders
from train import train_try
from qonnx.core.datatype import DataType
import torch.nn as nn

# ----- USER PARAS ------ #

def generate_ip(model_name, model, weight, activation, try_name= "/model_generation_test", folding_config_file = "auto"):
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


    build_dir = "./build" + try_name
    model_dir = "./model"
    data_dir = "./data"
    estimates_output_dir = "./estimates_output" + try_name
    rtlsim_output_dir = "./rtlsim_output" + try_name
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
    model.cpu()  # Ensure model is on CPU for export

    if model_name == 'unsw_fc':
    
        modified_model = deepcopy(model)
        W_orig = modified_model[0].weight.data.detach().numpy()
        W_orig.shape
        W_new = np.pad(W_orig, [(0,0), (0,7)])
        modified_model[0].weight.data = torch.from_numpy(W_new)
        print(modified_model[0].weight.shape)
      


        class CybSecMLPForExport(nn.Module):
            def __init__(self, my_pretrained_model):
                super(CybSecMLPForExport, self).__init__()
                self.pretrained = my_pretrained_model
                self.qnt_output = QuantIdentity(
                    quant_type='binary', 
                    scaling_impl_type='const',
                    bit_width=1, min_val=-1.0, max_val=1.0)
            
            def forward(self, x):
                # assume x contains bipolar {-1,1} elems
                # shift from {-1,1} -> {0,1} since that is the
                # input range for the trained network
                x = (x + torch.tensor([1.0]).to(x.device)) / 2.0  
                out_original = self.pretrained(x)
                out_final = self.qnt_output(out_original)   # output as {-1,1}     
                return out_final

        model_for_export = CybSecMLPForExport(modified_model)
        model_for_export.to(device)
        ready_model_filename = model_dir + "/" + model_ready_name

        input_shape = (1, 600)
        input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
        input_a = 2 * input_a - 1
        scale = 1.0
        input_t = torch.from_numpy(input_a * scale)
        #Move to CPU before export
        model_for_export.cpu()
        

        model_for_export.eval()

        # 导出前验证
        with torch.no_grad():
            out = model_for_export(input_t)
            print("Output range check - min: {}, max: {}".format(out.min(), out.max()))



        # Export to ONNX
        export_qonnx(
            model_for_export, export_path=ready_model_filename, input_t=input_t
        )

        # clean-up
        qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)
        print("Bipolar Model saved to %s" % ready_model_filename)
    else:
    # test model generation
        ready_model_filename = model_dir + "/" + model_ready_name
        input_shape = (1, 1, 32, 32)
        input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
        input_a = 2 * input_a - 1
        scale = 1.0
        input_t = torch.from_numpy(input_a * scale)
        #Move to CPU before export
        model.cpu()
        # Export to ONNX
        
        export_qonnx(
            model, export_path=ready_model_filename, input_t=input_t
        )
        # clean-up
        qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)


        print("Ready Model saved to %s" % ready_model_filename)

    # if os.path.exists(estimates_output_dir):
    #     shutil.rmtree(estimates_output_dir)
    #     print("Previous run results deleted!")

    if os.path.exists(rtlsim_output_dir):
        shutil.rmtree(rtlsim_output_dir)
        print("Previous run results deleted!")

    # if os.path.exists(tmp_path):
    #     shutil.rmtree(tmp_path)
    #     print("Previous run results deleted!")

    # if folding_config_file == "auto":
    #     cfg_stitched_ip = build.DataflowBuildConfig(
    #         output_dir          = rtlsim_output_dir,
    #         mvau_wwidth_max     = 10000,
    #         target_fps          = 1000000,
    #         synth_clk_period_ns = 10.0,    
    #         fpga_part           = pynq_part_map["U50"],
    #         # folding_config_file = folding_config_file,
    #         generate_outputs=[
    #             build_cfg.DataflowOutputType.STITCHED_IP,
    #             build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    #             build_cfg.DataflowOutputType.OOC_SYNTH,
    #         ]
    #     )
    # else:
    #     cfg_stitched_ip = build.DataflowBuildConfig(
    #         output_dir          = rtlsim_output_dir,
    #         mvau_wwidth_max     = 10000,
    #         target_fps          = 1000000,
    #         synth_clk_period_ns = 10.0,    
    #         fpga_part           = pynq_part_map["U50"],
    #         folding_config_file = folding_config_file,
    #         generate_outputs=[
    #             build_cfg.DataflowOutputType.STITCHED_IP,
    #             build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    #             build_cfg.DataflowOutputType.OOC_SYNTH,
    #         ]
    #     )
    if folding_config_file == "auto":
        cfg_stitched_ip = build.DataflowBuildConfig(
            output_dir          = rtlsim_output_dir,
            mvau_wwidth_max     = 80,
            target_fps          = 1000000,
            synth_clk_period_ns = 10.0,  
            fpga_part           = pynq_part_map["Pynq-Z2"],
            # folding_config_file = folding_config_file,
            generate_outputs=[
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                build_cfg.DataflowOutputType.OOC_SYNTH,
            ]
        )
    else:
        cfg_stitched_ip = build.DataflowBuildConfig(
            output_dir          = rtlsim_output_dir,
            mvau_wwidth_max     = 80,
            target_fps          = 1000000,
            synth_clk_period_ns = 10.0,    
            fpga_part           = pynq_part_map["Pynq-Z2"],
            folding_config_file = folding_config_file,
            generate_outputs=[
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                build_cfg.DataflowOutputType.OOC_SYNTH,
            ]
        )

    build.build_dataflow_cfg(ready_model_filename, cfg_stitched_ip)

if __name__ == "__main__":
    # model_name = '2c3f_relu'
    # weight = 4 
    # activation = 4
    # epochs = 500
    # model = train_try(model_name=model_name, w=weight, a=activation, epochs=epochs, random_seed=1998)
    # generate_ip(model_name='2c3f_relu', model = model, weight=4, activation=4, try_name="/test")
    model_name = 'unsw_fc'
    weight = 2 
    activation = 2
    epochs = 10
 
    model = get_model(model_name, weight, activation)
    model.load_state_dict(torch.load(f"./model/final_{model_name}_w{weight}_a{activation}_l1_pruned.pth"))
    # model.load_state_dict(torch.load(f"/home/changhong/prj/finn/script/EF_US/casestudy2/model/best_unsw_fc_w2_a2_1000.pth"))
    model.to(torch.device("cpu"))  # Ensure

    generate_ip(model_name=model_name, model = model, weight=2, activation=2, try_name="/test6")
