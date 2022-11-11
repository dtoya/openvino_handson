#!/bin/bash

wget https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/101-tensorflow-to-openvino/data/coco.jpg
wget https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/101-tensorflow-to-openvino/utils/imagenet_2012.txt
wget https://github.com/openvinotoolkit/openvino_notebooks/raw/main/notebooks/101-tensorflow-to-openvino/model/v3-small_224_1.0_float.pb

mo --input_model v3-small_224_1.0_float.pb -b 1 
