. ../venv/openvino/bin/activate
pot -q default -m public/yolo-v4-tf/FP16/yolo-v4-tf.xml -w public/yolo-v4-tf/FP16/yolo-v4-tf.bin --engine simplified --data-source images -d 
benchmark_app -m public/yolo-v4-tf/FP32/yolo-v4-tf.xml
benchmark_app -m results/optimized/model_name.xml
deactivate

