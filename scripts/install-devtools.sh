#!/bin/bash
python3 -m venv venv/openvino
. venv/openvino/bin/activate
python -m pip install --upgrade pip
pip install openvino
pip install opencv-python
pip install openvino-dev[tensorflow2,onnx,pytorch]
deactivate
