import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

import openvino.runtime as ov
from openvino.tools.mo import convert_model
import nncf

import numpy as np

import os
import re
import subprocess
from typing import List, Optional
import argparse

def validate(model: ov.CompiledModel,
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return (predictions == references).sum() / predictions.size

def run_benchmark(model_path: str, shape: Optional[List[int]] = None, verbose: bool = True) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))

def get_model_size(ir_path: str, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(ir_path.replace("xml", "bin"))
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} Mb")
        print(f"Model weights (bin): {bin_size:.3f} Mb")
        print(f"Model size:          {model_size:.3f} Mb")
    return model_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='cifar100_resnet56')
    parser.add_argument('--dataset_root', '-d', default='dataset')
    parser.add_argument('--output', '-o', default='result')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--max_drop', type=float, default=0.005)
    parser.add_argument('--subset_size', type=int, default=300)
    parser.add_argument('--download_only', default=False, action="store_true")
    args = parser.parse_args()

    mean = (0.507, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2761)
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ])

    dataset = CIFAR100(root=args.dataset_root+'/cifar100', train=False,
                        transform=data_transform, download=True)
    torch_model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.model,
                            pretrained=True, skip_validation=True)
    if args.download_only:
        return 

    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_file = args.output+'/'+args.model+'.onnx'
    torch.onnx.export(torch_model, dummy_input, onnx_file)

    def transform_fn(data_item):
        images, _ = data_item
        return images

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    core = ov.Core()
    fp32_model = convert_model(onnx_file)
    fp32_compiled_model = core.compile_model(model=fp32_model, device_name=args.device)
    fp32_ir_model = args.output+'/'+args.model+'_fp32.xml'
    ov.serialize(fp32_model, fp32_ir_model)

    int8_model = nncf.quantize(
        fp32_model,
        quantization_dataset,
        subset_size=args.subset_size
        )
    int8_compiled_model = core.compile_model(model=int8_model, device_name=args.device)
    int8_ir_model = args.output+'/'+args.model+'_int8.xml'
    ov.serialize(int8_model, int8_ir_model)

    int8ac_model = nncf.quantize_with_accuracy_control(
        fp32_model, 
        quantization_dataset, 
        quantization_dataset, 
        validation_fn=validate,
        max_drop=args.max_drop,
        subset_size=args.subset_size
        ) 
    int8ac_compiled_model = core.compile_model(model=int8ac_model, device_name=args.device)
    int8ac_ir_model = args.output+'/'+args.model+'_int8ac.xml'
    ov.serialize(int8ac_model, int8ac_ir_model)


    acc1_fp32 = validate(fp32_compiled_model, data_loader)
    acc1_int8 = validate(int8_compiled_model, data_loader)
    acc1_int8ac = validate(int8ac_compiled_model, data_loader)
    drop_int8 = (acc1_int8 - acc1_fp32) * 100 
    drop_int8ac = (acc1_int8ac - acc1_fp32) * 100 

    fps_fp32 = run_benchmark(fp32_ir_model, shape=[1, 3, 32, 32], verbose=True)
    fps_int8 = run_benchmark(int8_ir_model, shape=[1, 3, 32, 32], verbose=True)
    fps_int8ac = run_benchmark(int8ac_ir_model, shape=[1, 3, 32, 32], verbose=True)
    ratio_fps_int8 = fps_int8 / fps_fp32  
    ratio_fps_int8ac = fps_int8ac / fps_fp32  

    size_fp32 = get_model_size(fp32_ir_model, verbose=True) 
    size_int8 = get_model_size(int8_ir_model, verbose=True) 
    size_int8ac = get_model_size(int8ac_ir_model, verbose=True) 
    ratio_size_int8 = size_int8 / size_fp32  
    ratio_size_int8ac = size_int8ac / size_fp32  

    print("                    FP32     INT8(Basic)     INT8(Accuracy)   ")
    print("--------------------------------------------------------------")
    print(f"Accuracy [%]     : {acc1_fp32:7.3f} {acc1_int8:7.3f} ({drop_int8:5.1f}%) {acc1_int8ac:7.3f} ({drop_int8ac:5.1f}%)")
    print(f"Throughput [fps] : {fps_fp32:7.1f} {fps_int8:7.1f} ({ratio_fps_int8:5.1f})  {fps_int8ac:7.1f} ({ratio_fps_int8ac:5.1f})")
    print(f"Size [MB]        : {size_fp32:7.1f} {size_int8:7.1f} ({ratio_size_int8:5.1f})  {size_int8ac:7.1f} ({ratio_size_int8ac:5.1f})")

if __name__ == '__main__':

    main()

