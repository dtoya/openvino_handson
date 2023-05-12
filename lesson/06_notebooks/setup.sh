sudo apt-get update
#sudo apt-get upgrade
sudo apt-get -y install python3-venv build-essential python3-dev git-all
sudo apt-get -y install jupyter
python3 -m venv openvino_env
source openvino_env/bin/activate
git clone --depth=1 https://github.com/openvinotoolkit/openvino_notebooks.git
cd openvino_notebooks
python -m pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt
