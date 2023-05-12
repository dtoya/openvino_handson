. openvino_env/bin/activate
jupyter nbconvert --execute --to html openvino_notebooks/notebooks/001-hello-world/001-hello-world.ipynb --output-dir ./
#jupyter lab openvino_notebooks/notebooks
deactivate
