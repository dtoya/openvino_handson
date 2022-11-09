#!/bin/bash
# Install OpenVINO toolkit package
rm -rf $ARCHIVE_NAME.tgz* openvino_2022
ARCHIVE_NAME=l_openvino_toolkit_ubuntu20_2022.2.0.7713.af16ea1d79a_x86_64
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/linux/$ARCHIVE_NAME.tgz
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/linux/$ARCHIVE_NAME.tgz.sha256
sha256sum -c $ARCHIVE_NAME.tgz.sha256
tar xvzf $ARCHIVE_NAME.tgz 
mv $ARCHIVE_NAME openvino_2022
rm -rf $ARCHIVE_NAME.tgz* 


