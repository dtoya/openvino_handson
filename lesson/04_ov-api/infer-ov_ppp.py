# Reference: https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/101-tensorflow-to-openvino/101-tensorflow-to-openvino.ipynb
import time
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat

core = Core()
model = core.read_model("v3-small_224_1.0_float.xml")

# The MobileNet network expects images in RGB format
image_orig = cv2.imread(filename="coco.jpg")

# Transpose image to network input shape
input_tensor = np.expand_dims(image_orig, 0)

ppp = PrePostProcessor(model)
ppp.input().tensor() \
    .set_shape(input_tensor.shape) \
    .set_element_type(Type.u8) \
    .set_color_format(ColorFormat.BGR) \
    .set_layout(Layout('NHWC'))  
ppp.input().preprocess() \
    .convert_color(ColorFormat.RGB) \
    .resize(ResizeAlgorithm.RESIZE_LINEAR) \
    .convert_element_type().mean(127.5).scale(127.5)
ppp.input().model().set_layout(Layout('NHWC'))
ppp.output().tensor().set_element_type(Type.f32)
model = ppp.build()

compiled_model = core.compile_model(model=model, device_name="CPU")

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

result = compiled_model([input_tensor])[output_key]
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
    input_tensor = np.expand_dims(image_orig, 0)
    compiled_model([input_tensor])

end = time.perf_counter()
time_ir = end - start

print(
    f"IR model in Inference Engine/CPU: {time_ir/num_images:.4f} "
    f"seconds per image, FPS (includes pre-process): {num_images/time_ir:.2f}"
)

cv2.imshow('Image', image_orig)
cv2.waitKey()


