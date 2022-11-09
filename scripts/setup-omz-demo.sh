git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
sudo apt -y install python3-opencv build-essential libopencv-dev
python3 -m venv venv/omz_demo
. venv/omz_demo/bin/activate
pip install -U pip
pip install openvino
pip install -r open_model_zoo/demos/requirements.txt
pip install open_model_zoo/demos/common/python/
deactivate
. openvino_2022/setupvars.sh
chmod +x open_model_zoo/demos/build_demos.sh
open_model_zoo/demos/build_demos.sh -b=./omz_demos_build
