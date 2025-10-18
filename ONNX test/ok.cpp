#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include <onnxruntime_c_api.h>
}

void check_status(OrtStatus* status, const OrtApi* g_ort) {
    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

int main() {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env = NULL;
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env), g_ort);

    OrtSessionOptions* session_options = NULL;
    check_status(g_ort->CreateSessionOptions(&session_options), g_ort);

    const char* model_path = "/home/hiura/Documents/test/simple_cnn.onnx";
    OrtSession* session = NULL;
    check_status(g_ort->CreateSession(env, model_path, session_options, &session), g_ort);

    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);

    char* input_name = NULL;
    check_status(g_ort->SessionGetInputName(session, 0, allocator, &input_name), g_ort);

    const char* input_names[] = {input_name};
    const char* output_names[] = {"output"};

    OrtMemoryInfo* memory_info = NULL;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), g_ort);

    cv::Mat img = cv::imread("/home/hiura/Documents/test/mnist_digit.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        printf("Error: Cannot load image\n");
        return 1;
    }

    cv::resize(img, img, cv::Size(28, 28));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    const int64_t input_shape[] = {1, 1, 28, 28};
    float input_data[1 * 1 * 28 * 28];
    memcpy(input_data, img.data, sizeof(input_data));

    OrtValue* input_tensor = NULL;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_data, sizeof(input_data),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), g_ort);

    OrtValue* output_tensor = NULL;
    check_status(g_ort->Run(session, NULL,
                            input_names, (const OrtValue* const*)&input_tensor, 1,
                            output_names, 1, &output_tensor), g_ort);

    float* output_data = NULL;
    check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data), g_ort);

    int predicted_class = 0;
    float max_prob = output_data[0];
    for (int i = 1; i < 10; i++) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            predicted_class = i;
        }
    }

    printf("Predicted class: %d (prob = %.3f)\n", predicted_class, max_prob);

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseAllocator(allocator);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
