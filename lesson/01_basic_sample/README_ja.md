# OpenVINO Basic Sample

This sample demonstrates how to do inference of image classification models using Synchronous Inference Request API.  
Models with only one input and output are supported.

## How It Works

At startup, the sample application reads command line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin and performs synchronous inference. Then processes output data and write it to a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO™ Toolkit Samples guide.

## Running

```
hello_classification <path_to_model> <path_to_image> <device_name>
```

To run the sample, you need specify a model and image:

- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).
- you can use images from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - By default, OpenVINO™ Toolkit Samples and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).
>
> - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model using:

```
omz_downloader --name googlenet-v1
```

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

```
omz_converter --name googlenet-v1
```

4. Perform inference of `car.bmp` using the `googlenet-v1` model on a `GPU`, for example:

```
hello_classification googlenet-v1.xml car.bmp GPU
```

## Sample Output

The application outputs top-10 inference results.

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>
[ INFO ]
[ INFO ] Loading model files: /models/googlenet-v1.xml
[ INFO ] model name: GoogleNet
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: {1, 3, 224, 224}
[ INFO ]     outputs
[ INFO ]         output name: prob
[ INFO ]         output type: f32
[ INFO ]         output shape: {1, 1000}

Top 10 results:

Image /images/car.bmp

classid probability
------- -----------
656     0.8139648
654     0.0550537
468     0.0178375
436     0.0165405
705     0.0111694
817     0.0105820
581     0.0086823
575     0.0077515
734     0.0064468
785     0.0043983
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)

