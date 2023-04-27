import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

graph_def = tf.GraphDef()
graph_def.ParseFromString(open('v3-small_224_1.0_float.pb', 'rb').read())
tf.graph_util.import_graph_def(graph_def)
graph = tf.get_default_graph()

inputs  = [ n.name for n in graph_def.node if n.op in ('Placeholder')]
outputs = [ n.name for n in graph_def.node if n.op in ('Softmax')]
input_name  = '{}:0'.format(inputs[0])
output_name = '{}:0'.format(outputs[0])

# The MobileNet network expects images in RGB format
image_orig = cv2.imread(filename="coco.jpg")
image = cv2.cvtColor(image_orig, code=cv2.COLOR_BGR2RGB)

# Resize image to network input image shape
resized_image = cv2.resize(src=image, dsize=(224, 224))
resized_image = ( resized_image - 127.5 ) / 127.5

# Transpose image to network input shape
input_image = np.expand_dims(resized_image, 0)
output_tensor = graph.get_tensor_by_name(output_name)

sess = tf.Session(graph = graph)
result = sess.run(output_tensor, feed_dict = { input_name : input_image })
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
    result = sess.run(output_tensor, feed_dict = { input_name : input_image }) 

end = time.perf_counter()
time_ir = end - start
result_index = np.argmax(result)

print(
    f"TensorFlow/CPU: {time_ir/num_images:.4f} "
    f"seconds per image, FPS: {num_images/time_ir:.2f}"
)

cv2.imshow('Image', image_orig)
cv2.waitKey()


