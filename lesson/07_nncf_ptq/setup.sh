mkdir result
python3 -m venv venv
. venv/bin/activate
pip install -U pip 
pip install -r requirements.txt
python nncf_ptq_openvino_cifar100_resnet.py --download_only
