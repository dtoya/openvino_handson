git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
sudo apt -y install python3-opencv
python3 -m venv venv/omz_demo
. venv/omz_demo/bin/activate
pip install -U pip
pip install openvino==2022.3.0
pip install -r open_model_zoo/demos/requirements.txt
pip install open_model_zoo/demos/common/python/
deactivate
sudo apt -y install cmake build-essential libopencv-dev

rm -rf models
mkdir models
. ../../venv/openvino/bin/activate
omz_downloader --list models.lst -o models
omz_converter --list models.lst -o models -d models
deactivate

rm -rf data
mkdir data
wget -P data  https://storage.openvinotoolkit.org/data/test_data/videos/car-detection.mp4
wget -P data https://storage.openvinotoolkit.org/data/test_data/videos/face-demographics-walking.mp4 
wget -P data https://assets.amazon.science/ef/0b/234f82204da385f4893a150d7e34/sample01-orig.wav -O data/sample01_wind_noise.wav
wget -P data https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp 

. ../../openvino_2022/setupvars.sh
chmod +x open_model_zoo/demos/build_demos.sh
open_model_zoo/demos/build_demos.sh -b=./omz_demos_build
cp -r omz_demos_build/intel64/Release/ bin
