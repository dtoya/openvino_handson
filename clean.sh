#!/bin/bash
rm -rf venv openvino_2022 
cd lesson
for dir in `ls`;
do
    cd $dir; ./clean.sh; cd ..
done
cd ..
