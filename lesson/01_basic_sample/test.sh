. ../../venv/openvino/bin/activate
python ../../openvino_2022/samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP32/googlenet-v1.xml car_1.bmp CPU
deactivate

. ../../openvino_2022/setupvars.sh 
mkdir -p build
cd build
cmake ..
make
cd ..
./build/hello_classification public/googlenet-v1/FP32/googlenet-v1.xml car_1.bmp CPU

