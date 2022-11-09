#!/bin/bash
rm -rf data
mkdir data
wget -P data  https://storage.openvinotoolkit.org/data/test_data/videos/car-detection.mp4
wget -P data https://storage.openvinotoolkit.org/data/test_data/videos/face-demographics-walking.mp4 
wget -P data https://assets.amazon.science/ef/0b/234f82204da385f4893a150d7e34/sample01-orig.wav -O data/sample01_wind_noise.wav
wget -P data https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp 
