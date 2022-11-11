#!/bin/bash
rm -rf venv openvino_2022 
for dir in omz-demo ov-api pot;
do
    cd $dir; ./clean.sh; cd ..
done
