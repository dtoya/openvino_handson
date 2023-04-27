#!/bin/bash
sudo apt update
sudo apt -y install git python3-venv

rm -rf $ARCHIVE_NAME.tgz* openvino_2022
ARCHIVE_NAME=l_openvino_toolkit_ubuntu20_2022.3.0.9052.9752fafe8eb_x86_64
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/$ARCHIVE_NAME.tgz
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/$ARCHIVE_NAME.tgz.sha256
sha256sum -c $ARCHIVE_NAME.tgz.sha256
tar xvzf $ARCHIVE_NAME.tgz
mv $ARCHIVE_NAME openvino_2022
rm -rf $ARCHIVE_NAME.tgz*
sudo -E openvino_2022/install_dependencies/install_openvino_dependencies.sh

sudo apt update
sudo apt -y install curl
sudo -E openvino_2022/install_dependencies/install_NEO_OCL_driver.sh
sudo usermod -a -G video,render $USER

mkdir venv
sudo apt -y install python3-venv
python3 -m venv venv/openvino
. venv/openvino/bin/activate
python -m pip install --upgrade pip
pip install openvino
pip install opencv-python
pip install openvino-dev[tensorflow2,onnx,pytorch]
deactivate

cd lesson
for dir in `ls`;
do
    cd $dir; ./setup.sh; cd ..
done
cd ..

