# Reference: https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/101-tensorflow-to-openvino/101-tensorflow-to-openvino.ipynb
import time
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()
model = ie.read_model("v3-small_224_1.0_float.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

# The MobileNet network expects images in RGB format
image_orig = cv2.imread(filename="coco.jpg")
image = cv2.cvtColor(image_orig, code=cv2.COLOR_BGR2RGB)

# Resize image to network input image shape
resized_image = cv2.resize(src=image, dsize=(224, 224))
resized_image = ( resized_image - 127.5 ) / 127.5

# Transpose image to network input shape
input_image = np.expand_dims(resized_image, 0)

result = compiled_model([input_image])[output_key]
result = np.squeeze(result)

result_index = np.argmax(result)

# Convert the inference result to a class name.
imagenet_classes = open("imagenet_2012.txt").read().splitlines()

# The model description states that for this model, class 0 is background,
# so we add background at the beginning of imagenet_classes
imagenet_classes = ['background'] + imagenet_classes

print(f"Result: {imagenet_classes[result_index]} {result[result_index]}")

num_images = 1000

start = time.perf_counter()

for _ in range(num_images):
    image = cv2.cvtColor(image_orig, code=cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(src=image, dsize=(224, 224))
    resized_image = ( resized_image - 127.5 ) / 127.5
    input_image = np.expand_dims(resized_image, 0)
    compiled_model([input_image])

end = time.perf_counter()
time_ir = end - start

print(
    f"IR model in Inference Engine/CPU: {time_ir/num_images:.4f} "
    f"seconds per image, FPS (includes pre-process): {num_images/time_ir:.2f}"
)

cv2.imshow('Image', image_orig)
cv2.waitKey()


