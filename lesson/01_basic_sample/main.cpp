// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

//template <class T>
void topResults(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
    ov::Shape shape = input.get_shape();
    size_t input_rank = shape.size();
    OPENVINO_ASSERT(input_rank != 0 && shape[0] != 0, "Input tensor has incorrect dimensions!");
    size_t batchSize = shape[0];
    std::vector<unsigned> indexes(input.get_size() / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.get_size()));
    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        const size_t offset = i * (input.get_size() / batchSize);
        //const T* batchData = input.data<const T>();
        const float* batchData = input.data<const float>();
        batchData += offset;

        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(std::begin(indexes),
                            std::begin(indexes) + n,
                            std::end(indexes),
                            [&batchData](unsigned l, unsigned r) {
                                return batchData[l] > batchData[r];
                            });
        for (unsigned j = 0; j < n; j++) {
            output.at(i * n + j) = indexes.at(j);
        }
    }
}


int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 4) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const std::string args{argv[0]};
        const std::string model_path{argv[1]};
        const std::string image_path{argv[2]};
        const std::string device_name{argv[3]};

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        std::cout << "Loading model files: " << model_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Set up input

        cv::Mat image = cv::imread(image_path.c_str());

        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, (long unsigned int) image.rows, (long unsigned int) image.cols, 3};
        //std::shared_ptr<unsigned char> input_data = image.data;

        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, image.data);

        const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------

        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor`
        // - layout of data is 'NHWC'
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 4) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");
        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        ov::CompiledModel compiled_model = core.compile_model(model, device_name);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();

        std::vector<unsigned> results;
        size_t nTop = 10;
        topResults((unsigned int)nTop, output_tensor, results);
        std::cout << std::endl << "Top " << nTop << " results:" << std::endl;
        
        for (size_t id = 0, cnt = 0; id < nTop; ++cnt, ++id) {
            const auto index = results.at(id);
            const auto result = output_tensor.data<const float>()[index];
            std::cout << id << " : " << index << " : " << result << std::endl;
        }
        std::cout << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
