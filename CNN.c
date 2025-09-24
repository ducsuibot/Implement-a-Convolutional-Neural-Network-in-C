#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- Tham số cố định ----------------
// Kích thước ảnh input: 28x28
// Conv1: 2 filter, kernel 5x5, output size 24x24
// Pool1: 2x2 stride 2 -> 12x12
// Conv2: 4 filter, kernel 3x3, in_channels = 2, output size 10x10
// Pool2: 2x2 stride 2 -> 5x5
// Flatten: 4*5*5 = 100
// Output classes: 10

// ---------------- Utility inline ----------------
static inline float frand_init(float s) {
    return s * ((float)rand() / (float)RAND_MAX * 2.f - 1.f);
}

static inline float relu(float x){ return x > 0.f ? x : 0.f; }
static inline float relu_deriv(float x){ return x > 0.f ? 1.f : 0.f; }
static inline float sigmoid(float x){ return 1.f / (1.f + expf(-x)); }
static inline float sigmoid_deriv_from_y(float y){ return y * (1.f - y); }

void stable_softmax(const float *z, float *p, int n){
    float maxz = z[0];
    for(int i=1;i<n;i++) if(z[i]>maxz) maxz=z[i];
    float sum=0.f;
    for(int i=0;i<n;i++){ p[i] = expf(z[i]-maxz); sum += p[i]; }
    for(int i=0;i<n;i++) p[i] /= sum;
}

float cross_entropy_loss(const float *p, int y){
    float eps = 1e-12f;
    return -logf(p[y] + eps);
}

// ---------------- Structures tham số và gradient ----------------
typedef struct {
    // Conv1: 2 filters, 5x5 each
    float conv1_weight[2][5][5];
    float conv1_bias[2];
    // Conv2: 4 filters, in_channels=2, 3x3 each
    float conv2_weight[4][2][3][3];
    float conv2_bias[4];
    // Dense: output 10, input 100 (flatten)
    float fc_weight[10][100];
    float fc_bias[10];
} Parameters;

typedef struct {
    float conv1_weight[2][5][5];
    float conv1_bias[2];
    float conv2_weight[4][2][3][3];
    float conv2_bias[4];
    float fc_weight[10][100];
    float fc_bias[10];
} Gradients;

// ---------------- Cache để lưu các giá trị trong forward phục vụ backprop ----------------
typedef struct {
    float input_image[28][28];
    float conv1_preact[2][24][24];
    float conv1_relu[2][24][24];
    float pool1_out[2][12][12];
    int   pool1_argmax_idx[2][12][12];
    float conv2_preact[4][10][10];
    float conv2_sigmoid[4][10][10];
    float pool2_out[4][5][5];
    int   pool2_argmax_idx[4][5][5];
    float flatten[100];
    float logits[10];
    float probabilities[10];
    int label;
} ForwardCache;

// ---------------- Forward functions ----------------
void conv1_forward(const float input[28][28], const Parameters *params, ForwardCache *cache){
    for(int k=0;k<2;k++){
        for(int i=0;i<24;i++){
            for(int j=0;j<24;j++){
                float s = params->conv1_bias[k];
                for(int u=0;u<5;u++){
                    for(int v=0;v<5;v++){
                        s += params->conv1_weight[k][u][v] * input[i+u][j+v];
                    }
                }
                cache->conv1_preact[k][i][j] = s;
                cache->conv1_relu[k][i][j] = relu(s);
            }
        }
    }
}

void maxpool2x2_forward_pool1(ForwardCache *cache){
    for(int k=0;k<2;k++){
        for(int i=0;i<12;i++){
            for(int j=0;j<12;j++){
                float max = -INFINITY; int max_idx = 0;
                for(int u=0; u<2; u++){
                    for(int v=0; v<2; v++){
                        float val = cache->conv1_relu[k][i*2 + u][j*2 + v];
                        int idx = u*2 + v;
                        if(val > max){ max = val; max_idx = idx; }
                    }
                }
                cache->pool1_out[k][i][j] = max;
                cache->pool1_argmax_idx[k][i][j] = max_idx;
            }
        }
    }
}

void conv2_forward(const ForwardCache *cache, const Parameters *params){
    for(int k=0;k<4;k++){
        for(int i=0;i<10;i++){
            for(int j=0;j<10;j++){
                float s = params->conv2_bias[k];
                for(int c=0;c<2;c++){
                    for(int u=0;u<3;u++){
                        for(int v=0;v<3;v++){
                            s += params->conv2_weight[k][c][u][v] * cache->pool1_out[c][i+u][j+v];
                        }
                    }
                }
                // lưu preact và sigmoid
                ((ForwardCache*)cache)->conv2_preact[k][i][j] = s;
                ((ForwardCache*)cache)->conv2_sigmoid[k][i][j] = sigmoid(s);
            }
        }
    }
}

void maxpool2x2_forward_pool2(ForwardCache *cache){
    for(int k=0;k<4;k++){
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                float max = -INFINITY; int max_idx = 0;
                for(int u=0; u<2; u++){
                    for(int v=0; v<2; v++){        
                        float val = cache->conv2_sigmoid[k][i*2+u][j*2+v];
                        int idx = u*2 + v;
                        if(val > max){ max = val; max_idx = idx; }
                    }
                }
                cache->pool2_out[k][i][j] = max;
                cache->pool2_argmax_idx[k][i][j] = max_idx;
            }
        }
    }
}

void flatten_forward(ForwardCache *cache){
    int t=0;
    for(int k=0;k<4;k++){
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                cache->flatten[t++] = cache->pool2_out[k][i][j];
            }
        }
    }
}

void dense_forward(ForwardCache *cache, const Parameters *params){
    for(int i=0;i<10;i++){
        float s = params->fc_bias[i];
        for(int j=0;j<100;j++) s += params->fc_weight[i][j] * cache->flatten[j];
        cache->logits[i] = s;
    }
    stable_softmax(cache->logits, cache->probabilities, 10);
}

float forward_pass(const float input[28][28], int label, const Parameters *params, ForwardCache *cache){
    // copy input
    for(int i=0;i<28;i++) for(int j=0;j<28;j++) cache->input_image[i][j] = input[i][j];
    cache->label = label;

    conv1_forward(cache->input_image, params, cache);
    maxpool2x2_forward_pool1(cache);
    conv2_forward(cache, params);
    maxpool2x2_forward_pool2(cache);
    flatten_forward(cache);
    dense_forward(cache, params);
    return cross_entropy_loss(cache->probabilities, label);
}

// ---------------- Backward functions ----------------
void dense_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, float dflatten[100]){
    float dlogits[10];
    for(int i=0;i<10;i++) dlogits[i] = cache->probabilities[i] - (i==cache->label ? 1.f : 0.f);
    memset(grads->fc_weight, 0, sizeof(grads->fc_weight));
    memset(grads->fc_bias, 0, sizeof(grads->fc_bias));
    for(int i=0;i<10;i++){
        grads->fc_bias[i] += dlogits[i];
        for(int j=0;j<100;j++) grads->fc_weight[i][j] += dlogits[i] * cache->flatten[j];
    }

    for(int j=0;j<100;j++){
        float s=0.f;
        for(int i=0;i<10;i++) s += params->fc_weight[i][j] * dlogits[i];
        dflatten[j] = s;
    }
}

void unflatten_backward_to_pool2(const float dflatten[100], float dpool2[4][5][5]){
    int t=0;
    for(int k=0;k<4;k++){
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                dpool2[k][i][j] = dflatten[t++];
            }
        }
    }
}

void maxpool2x2_backward_pool2(const ForwardCache *cache, const float dpool2[4][5][5], float dconv2_sig[4][10][10]){
    // zero init
    for(int k=0;k<4;k++) for(int i=0;i<10;i++) for(int j=0;j<10;j++) dconv2_sig[k][i][j] = 0.f;

    for(int k=0;k<4;k++){
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                int idx = cache->pool2_argmax_idx[k][i][j];
                int u = idx/2, v = idx%2;
                int si = i*2 + u, sj = j*2 + v;
                dconv2_sig[k][si][sj] += dpool2[k][i][j];
            }
        }
    }
}

void sigmoid_backward(const ForwardCache *cache, const float dconv2_sig[4][10][10], float dconv2[4][10][10]){
    for(int k=0;k<4;k++){
        for(int i=0;i<10;i++){
            for(int j=0;j<10;j++){
                float y = cache->conv2_sigmoid[k][i][j];
                dconv2[k][i][j] = dconv2_sig[k][i][j] * sigmoid_deriv_from_y(y);
            }
        }
    }
}

void conv2_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads,
                    const float dconv2[4][10][10],
                    float dpool1[2][12][12]){
    // zero grads
    memset(grads->conv2_weight, 0, sizeof(grads->conv2_weight));
    memset(grads->conv2_bias, 0, sizeof(grads->conv2_bias));
    // zero dpool1
    for(int c=0;c<2;c++) for(int i=0;i<12;i++) for(int j=0;j<12;j++) dpool1[c][i][j] = 0.f;

    // dW and dB
    for(int k=0;k<4;k++){
        for(int i=0;i<10;i++){
            for(int j=0;j<10;j++){
                float g = dconv2[k][i][j];
                grads->conv2_bias[k] += g;
                for(int c=0;c<2;c++){
                    for(int u=0;u<3;u++){
                        for(int v=0;v<3;v++){
                            grads->conv2_weight[k][c][u][v] += g * cache->pool1_out[c][i+u][j+v];
                        }
                    }
                }
            }
        }
    }

    // backprop to pool1 (valid conv backprop)
    for(int c=0;c<2;c++){
        for(int i=0;i<12;i++){
            for(int j=0;j<12;j++){
                float s = 0.f;
                for(int k=0;k<4;k++){
                    for(int u=0;u<3;u++){
                        for(int v=0;v<3;v++){
                            int ip = i - u; int jp = j - v;
                            if(ip>=0 && ip<10 && jp>=0 && jp<10){
                                s += params->conv2_weight[k][c][u][v] * dconv2[k][ip][jp];
                            }
                        }
                    }
                }
                dpool1[c][i][j] = s;
            }
        }
    }
}

void maxpool2x2_backward_pool1(const ForwardCache *cache, const float dpool1[2][12][12], float dconv1_relu[2][24][24]){
    // zero init
    for(int k=0;k<2;k++) for(int i=0;i<24;i++) for(int j=0;j<24;j++) dconv1_relu[k][i][j] = 0.f;

    for(int k=0;k<2;k++){
        for(int i=0;i<12;i++){
            for(int j=0;j<12;j++){
                int idx = cache->pool1_argmax_idx[k][i][j];
                int di = idx/2, dj = idx%2;
                int si = i*2 + di, sj = j*2 + dj;
                dconv1_relu[k][si][sj] += dpool1[k][i][j];
            }
        }
    }
}

void relu_backward_conv1(const ForwardCache *cache, const float dconv1_relu[2][24][24], float dconv1[2][24][24]){
    for(int k=0;k<2;k++){
        for(int i=0;i<24;i++){
            for(int j=0;j<24;j++){
                float g = dconv1_relu[k][i][j];
                dconv1[k][i][j] = g * relu_deriv(cache->conv1_preact[k][i][j]);
            }
        }
    }
}

void conv1_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads,
                    const float dconv1[2][24][24]){
    // zero grads
    memset(grads->conv1_weight, 0, sizeof(grads->conv1_weight));
    memset(grads->conv1_bias, 0, sizeof(grads->conv1_bias));

    for(int k=0;k<2;k++){
        for(int i=0;i<24;i++){
            for(int j=0;j<24;j++){
                float g = dconv1[k][i][j];
                grads->conv1_bias[k] += g;
                for(int u=0;u<5;u++){
                    for(int v=0;v<5;v++){
                        grads->conv1_weight[k][u][v] += g * cache->input_image[i+u][j+v];
                    }
                }
            }
        }
    }
}

// ---------------- Helpers: zero grads, sgd update, init params ----------------
void zero_grads(Gradients *gr){
    memset(gr, 0, sizeof(*gr));
}

void sgd_update(Parameters *params, const Gradients *grads, float learning_rate){
    for(int k=0;k<2;k++){
        for(int u=0;u<5;u++) for(int v=0;v<5;v++) params->conv1_weight[k][u][v] -= learning_rate * grads->conv1_weight[k][u][v];
        params->conv1_bias[k] -= learning_rate * grads->conv1_bias[k];
    }
    for(int k=0;k<4;k++){
        for(int c=0;c<2;c++) for(int u=0;u<3;u++) for(int v=0;v<3;v++) params->conv2_weight[k][c][u][v] -= learning_rate * grads->conv2_weight[k][c][u][v];
        params->conv2_bias[k] -= learning_rate * grads->conv2_bias[k];
    }
    for(int i=0;i<10;i++){
        for(int j=0;j<100;j++) params->fc_weight[i][j] -= learning_rate * grads->fc_weight[i][j];
        params->fc_bias[i] -= learning_rate * grads->fc_bias[i];
    }
}

void init_parameters(Parameters *params){
    srand(42);
    float s1 = 0.1f, s2 = 0.1f, s3 = 0.1f;
    for(int k=0;k<2;k++){
        for(int u=0;u<5;u++) for(int v=0;v<5;v++) params->conv1_weight[k][u][v] = frand_init(s1);
        params->conv1_bias[k] = 0.f;
    }
    for(int k=0;k<4;k++){
        for(int c=0;c<2;c++) for(int u=0;u<3;u++) for(int v=0;v<3;v++) params->conv2_weight[k][c][u][v] = frand_init(s2);
        params->conv2_bias[k] = 0.f;
    }
    for(int o=0;o<10;o++){
        for(int i=0;i<100;i++) params->fc_weight[o][i] = frand_init(s3);
        params->fc_bias[o] = 0.f;
    }
}

// ---------------- MNIST IDX loader (như trước) ----------------
static uint32_t read_be_u32(FILE *f){
    unsigned char b[4];
    if (fread(b,1,4,f)!=4){ perror("read_u32"); exit(1); }
    return ((uint32_t)b[0]<<24) | ((uint32_t)b[1]<<16) | ((uint32_t)b[2]<<8) | ((uint32_t)b[3]);
}

static void load_mnist_images(const char *path, float (*imgs)[28][28], int expected_count){
    FILE *f = fopen(path, "rb");
    if(!f){ perror(path); exit(1); }
    uint32_t magic = read_be_u32(f);
    uint32_t n     = read_be_u32(f);
    uint32_t h     = read_be_u32(f);
    uint32_t w     = read_be_u32(f);
    if(magic != 2051 || h!=28 || w!=28){
        fprintf(stderr, "MNIST images header mismatch: magic=%u, n=%u, h=%u, w=%u\n", magic, n, h, w);
        exit(1);
    }
    if(expected_count>0 && (int)n!=expected_count){
        fprintf(stderr, "Expected %d images, but file has %u\n", expected_count, n);
        exit(1);
    }
    size_t one = (size_t)h*w;
    unsigned char *buf = (unsigned char*)malloc(one);
    if(!buf){ perror("malloc img buf"); exit(1); }
    for(uint32_t i=0;i<n;i++){
        if(fread(buf,1,one,f)!=one){ fprintf(stderr,"Short read img %u\n", i); exit(1); }
        for(int r=0;r<28;r++){
            for(int c=0;c<28;c++){
                imgs[i][r][c] = (float)buf[r*28+c] / 255.0f; // normalize to [0,1]
            }
        }
    }
    free(buf);
    fclose(f);
}

static void load_mnist_labels(const char *path, unsigned char *labels, int expected_count){
    FILE *f = fopen(path, "rb");
    if(!f){ perror(path); exit(1); }
    uint32_t magic = read_be_u32(f);
    uint32_t n     = read_be_u32(f);
    if(magic != 2049){
        fprintf(stderr, "MNIST labels header mismatch: magic=%u, n=%u\n", magic, n);
        exit(1);
    }
    if(expected_count>0 && (int)n!=expected_count){
        fprintf(stderr, "Expected %d labels, but file has %u\n", expected_count, n);
        exit(1);
    }
    if(fread(labels,1,n,f)!=n){ fprintf(stderr,"Short read labels\n"); exit(1); }
    fclose(f);
}

// ---------------- Backward pass orchestration ----------------
void backward_pass(const Parameters *params, const ForwardCache *cache, Gradients *grads){
    zero_grads(grads);

    float dflatten[100];
    dense_backward(cache, params, grads, dflatten);

    float dpool2[4][5][5];
    unflatten_backward_to_pool2(dflatten, dpool2);

    float dconv2_sig[4][10][10];
    maxpool2x2_backward_pool2(cache, dpool2, dconv2_sig);

    float dconv2[4][10][10];
    sigmoid_backward(cache, dconv2_sig, dconv2);

    float dpool1[2][12][12];
    conv2_backward(cache, params, grads, dconv2, dpool1);

    float dconv1_relu[2][24][24];
    maxpool2x2_backward_pool1(cache, dpool1, dconv1_relu);

    float dconv1[2][24][24];
    relu_backward_conv1(cache, dconv1_relu, dconv1);

    conv1_backward(cache, params, grads, dconv1);
}

// ---------------- Demo main (mô phỏng huấn luyện 1 epoch full dataset) ----------------
#ifndef TRAIN_IMAGE
#define TRAIN_IMAGE "E:/CNN/train-images.idx3-ubyte"
#define TRAIN_LABEL "E:/CNN/train-labels.idx1-ubyte"
#define TEST_IMAGE  "E:/CNN/t10k-images.idx3-ubyte"
#define TEST_LABEL  "E:/CNN/t10k-labels.idx1-ubyte"
#endif

int main(){
    // Load MNIST (như cũ)
    static float train_images[60000][28][28];
    static unsigned char train_labels[60000];
    static float test_images[10000][28][28];
    static unsigned char test_labels[10000];

    printf("Loading MNIST from %s, %s, %s, %s ...\n", TRAIN_IMAGE, TRAIN_LABEL, TEST_IMAGE, TEST_LABEL);
    load_mnist_images(TRAIN_IMAGE, train_images, 60000);
    load_mnist_labels(TRAIN_LABEL, train_labels, 60000);
    load_mnist_images(TEST_IMAGE,  test_images,  10000);
    load_mnist_labels(TEST_LABEL,  test_labels,  10000);
    printf("Loaded.\n");

    // Init params
    Parameters params; init_parameters(&params);
    ForwardCache cache;
    Gradients grads;

    // Hyperparameters
    int epochs = 5;
    float learning_rate = 0.01f;

    for(int epoch=0; epoch<epochs; epoch++){
        float loss = 0.f;
        int correct = 0;
        for(int i=0;i<60000;i++){
            int label = (int)train_labels[i];
            loss += forward_pass(train_images[i], label, &params, &cache);

            int predict = 0;
            for(int j=1;j<10;j++) if(cache.probabilities[j] > cache.probabilities[predict]) predict = j;
            if(predict == label) correct++;

            backward_pass(&params, &cache, &grads);
            sgd_update(&params, &grads, learning_rate);
        }
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%%\n", epoch+1, loss/60000.0f, 100.0f * (float)correct / 60000.0f);

        // Evaluate on test set
        float test_loss = 0.f;
        int test_correct = 0;
        for(int i=0;i<10000;i++){
            int label = (int)test_labels[i];
            test_loss += forward_pass(test_images[i], label, &params, &cache);
            int predict = 0;
            for(int j=1;j<10;j++) if(cache.probabilities[j] > cache.probabilities[predict]) predict = j;
            if(predict == label) test_correct++;
        }
        printf("  -> Test  - Loss: %.4f - Accuracy: %.2f%%\n", test_loss/10000.0f, 100.0f * (float)test_correct / 10000.0f);
    }

    return 0;
}
