. ../venv/openvino/bin/activate
mo --input_model v3-small_224_1.0_float.pb -b 1
python infer-tf.py
python infer-ov.py
python infer-ov_ppp.py
python infer-ov_ppp_ireq.py
python infer-ov_async.py
deactivate
