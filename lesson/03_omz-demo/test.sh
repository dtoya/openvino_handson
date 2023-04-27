
. ../../openvino_2022/setupvars.sh
chmod +x open_model_zoo/demos/build_demos.sh
open_model_zoo/demos/build_demos.sh -b=./omz_demos_build
cp -r omz_demos_build/intel64/Release/ bin

. venv/omz_demo/bin/activate
scripts/run_object_detection_demo_python.sh
scripts/run_object_detection_demo_cpp.sh
scripts/run_human_pose_estimation_demo_python.sh
scripts/run_human_pose_estimation_demo_cpp.sh
scripts/run_image_restoration_demo.sh       
scripts/run_image_super_resolution_demo.sh  
scripts/run_noise_suppression_demo.sh       
deactivate

