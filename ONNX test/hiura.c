#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>

void check_status(OrtStatus* status, const OrtApi* g_ort) {
    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

// Doc du lieu anh tu file (28x28 float)
int read_input_data(const char* filename, float* buffer, size_t size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }

    for (size_t i = 0; i < size; i++) {
        if (fscanf(f, "%f", &buffer[i]) != 1) {
            printf("Error: Invalid data format at index %zu\n", i);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

int main() {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env = NULL;
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env), g_ort);

    OrtSessionOptions* session_options = NULL;
    check_status(g_ort->CreateSessionOptions(&session_options), g_ort);

    const char* model_path = "/home/hiura/Documents/test/lenet5_mnist.onnx";
    OrtSession* session = NULL;
    check_status(g_ort->CreateSession(env, model_path, session_options, &session), g_ort);

    OrtAllocator* allocator = NULL;
    check_status(g_ort->GetAllocatorWithDefaultOptions(&allocator), g_ort);

    char* input_name = NULL;
    check_status(g_ort->SessionGetInputName(session, 0, allocator, &input_name), g_ort);

    const char* input_names[] = {input_name};
    const char* output_names[] = {"output"};

    OrtMemoryInfo* memory_info = NULL;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), g_ort);

    // Doc du lieu anh tu file
    const char* input_file = "/home/hiura/Documents/test/mnist_input.txt";
    float input_data[1 * 1 * 28 * 28];
    if (!read_input_data(input_file, input_data, 28 * 28)) {
        printf("Error reading input data\n");
        return 1;
    }

    const int64_t input_shape[] = {1, 1, 28, 28};

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

    // --- Tinh Softmax ---
    float exp_sum = 0.0f;
    float probs[10];
    for (int i = 0; i < 10; i++) exp_sum += expf(output_data[i]);
    for (int i = 0; i < 10; i++) probs[i] = expf(output_data[i]) / exp_sum;

    // --- In tat ca output ---
    printf("\nOutput logits va xac suat:\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %d: logit = %.3f | prob = %.5f\n", i, output_data[i], probs[i]);
    }

    // --- Tim lop co xac suat cao nhat ---
    int predicted_class = 0;
    float max_prob = probs[0];
    for (int i = 1; i < 10; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted_class = i;
        }
    }

    printf("\nPredicted class: %d (prob = %.5f)\n", predicted_class, max_prob);

    // --- Giai phong bo nho ---
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    // KHONG giai phong allocator mac dinh
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
