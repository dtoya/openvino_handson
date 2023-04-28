. ../../venv/openvino/bin/activate
benchmark_app -m public/resnet-50-tf/FP32/resnet-50-tf.xml -d CPU 
benchmark_app -m public/resnet-50-tf/FP32/resnet-50-tf.xml -d GPU 
benchmark_app -m intel/person-detection-0303/FP32/person-detection-0303.xml
benchmark_app -m intel/person-detection-0303/FP16-INT8/person-detection-0303.xml
deactivate
