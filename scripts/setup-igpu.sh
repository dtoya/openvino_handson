#!/bin/bash
# Install GPU driver
sudo apt update
sudo apt -y install curl
sudo -E openvino_2022/install_dependencies/install_NEO_OCL_driver.sh
sudo usermod -a -G video,render $USER


