#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- LeNet-5 Architecture Parameters ----------------
// Input: 32x32 (MNIST 28x28 padded with zeros)
// C1: 6 filters, 5x5, output 28x28x6
// S2: 2x2 avg pool, stride 2, output 14x14x6
// C3: 16 filters, 5x5, in_channels=6, output 10x10x16
// S4: 2x2 avg pool, stride 2, output 5x5x16
// C5: 120 filters, 5x5, in_channels=16, output 1x1x120
// F6: 120 inputs, 84 outputs, tanh activation
// Output: 84 inputs, 10 outputs, softmax

// ---------------- Utility inline ----------------
static inline float frand_init(float s) {
    return s * ((float)rand() / (float)RAND_MAX * 2.f - 1.f);
}

static inline float tanh_act(float x) { return tanhf(x); }
static inline float tanh_deriv(float x) { return 1.f - tanhf(x) * tanhf(x); }
static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
static inline float sigmoid_deriv_from_y(float y) { return y * (1.f - y); }

void stable_softmax(const float *z, float *p, int n) {
    float maxz = z[0];
    for (int i = 1; i < n; i++) if (z[i] > maxz) maxz = z[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) { p[i] = expf(z[i] - maxz); sum += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= sum;
}

float cross_entropy_loss(const float *p, int y) {
    float eps = 1e-12f;
    return -logf(p[y] + eps);
}

// ---------------- Parameter and Gradient Structures ----------------
typedef struct {
    float c1_weight[6][5][5];        // C1: 6 filters, 5x5
    float c1_bias[6];
    float c3_weight[16][6][5][5];    // C3: 16 filters, in_channels=6, 5x5
    float c3_bias[16];
    float c5_weight[120][16][5][5];  // C5: 120 filters, in_channels=16, 5x5
    float c5_bias[120];
    float f6_weight[84][120];        // F6: 120 inputs, 84 outputs
    float f6_bias[84];
    float output_weight[10][84];     // Output: 84 inputs, 10 outputs
    float output_bias[10];
} Parameters;

typedef struct {
    float c1_weight[6][5][5];
    float c1_bias[6];
    float c3_weight[16][6][5][5];
    float c3_bias[16];
    float c5_weight[120][16][5][5];
    float c5_bias[120];
    float f6_weight[84][120];
    float f6_bias[84];
    float output_weight[10][84];
    float output_bias[10];
} Gradients;

// ---------------- Forward Cache ----------------
typedef struct {
    float input_image[32][32];
    float c1_preact[6][28][28];
    float c1_tanh[6][28][28];
    float s2_out[6][14][14];
    float c3_preact[16][10][10];
    float c3_tanh[16][10][10];
    float s4_out[16][5][5];
    float c5_preact[120][1][1];
    float c5_tanh[120];
    float f6_preact[84];
    float f6_tanh[84];
    float logits[10];
    float probabilities[10];
    int label;
} ForwardCache;

// ---------------- Forward Functions ----------------
void c1_forward(const float input[32][32], const Parameters *params, ForwardCache *cache) {
    for (int k = 0; k < 6; k++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                float s = params->c1_bias[k];
                for (int u = 0; u < 5; u++) {
                    for (int v = 0; v < 5; v++) {
                        s += params->c1_weight[k][u][v] * input[i + u][j + v];
                    }
                }
                cache->c1_preact[k][i][j] = s;
                cache->c1_tanh[k][i][j] = tanh_act(s);
            }
        }
    }
}

void s2_avgpool_forward(const ForwardCache *cache) {
    for (int k = 0; k < 6; k++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                float sum = 0.f;
                for (int u = 0; u < 2; u++) {
                    for (int v = 0; v < 2; v++) {
                        sum += cache->c1_tanh[k][i * 2 + u][j * 2 + v];
                    }
                }
                ((ForwardCache*)cache)->s2_out[k][i][j] = sum / 4.f;
            }
        }
    }
}

void c3_forward(const ForwardCache *cache, const Parameters *params) {
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                float s = params->c3_bias[k];
                for (int c = 0; c < 6; c++) {
                    for (int u = 0; u < 5; u++) {
                        for (int v = 0; v < 5; v++) {
                            s += params->c3_weight[k][c][u][v] * cache->s2_out[c][i + u][j + v];
                        }
                    }
                }
                ((ForwardCache*)cache)->c3_preact[k][i][j] = s;
                ((ForwardCache*)cache)->c3_tanh[k][i][j] = tanh_act(s);
            }
        }
    }
}

void s4_avgpool_forward(const ForwardCache *cache) {
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float sum = 0.f;
                for (int u = 0; u < 2; u++) {
                    for (int v = 0; v < 2; v++) {
                        sum += cache->c3_tanh[k][i * 2 + u][j * 2 + v];
                    }
                }
                ((ForwardCache*)cache)->s4_out[k][i][j] = sum / 4.f;
            }
        }
    }
}

void c5_forward(const ForwardCache *cache, const Parameters *params) {
    for (int k = 0; k < 120; k++) {
        float s = params->c5_bias[k];
        for (int c = 0; c < 16; c++) {
            for (int u = 0; u < 5; u++) {
                for (int v = 0; v < 5; v++) {
                    s += params->c5_weight[k][c][u][v] * cache->s4_out[c][u][v];
                }
            }
        }
        ((ForwardCache*)cache)->c5_preact[k][0][0] = s;
        ((ForwardCache*)cache)->c5_tanh[k] = tanh_act(s);
    }
}

void f6_forward(const ForwardCache *cache, const Parameters *params) {
    for (int i = 0; i < 84; i++) {
        float s = params->f6_bias[i];
        for (int j = 0; j < 120; j++) {
            s += params->f6_weight[i][j] * cache->c5_tanh[j];
        }
        ((ForwardCache*)cache)->f6_preact[i] = s;
        ((ForwardCache*)cache)->f6_tanh[i] = tanh_act(s);
    }
}

void output_forward(const ForwardCache *cache, const Parameters *params) {
    for (int i = 0; i < 10; i++) {
        float s = params->output_bias[i];
        for (int j = 0; j < 84; j++) {
            s += params->output_weight[i][j] * cache->f6_tanh[j];
        }
        ((ForwardCache*)cache)->logits[i] = s;
    }
    stable_softmax(cache->logits, ((ForwardCache*)cache)->probabilities, 10);
}

float forward_pass(const float input[32][32], int label, const Parameters *params, ForwardCache *cache) {
    for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) cache->input_image[i][j] = input[i][j];
    cache->label = label;

    c1_forward(cache->input_image, params, cache);
    s2_avgpool_forward(cache);
    c3_forward(cache, params);
    s4_avgpool_forward(cache);
    c5_forward(cache, params);
    f6_forward(cache, params);
    output_forward(cache, params);
    return cross_entropy_loss(cache->probabilities, label);
}

// ---------------- Backward Functions ----------------
void output_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, float df6_tanh[84]) {
    float dlogits[10];
    for (int i = 0; i < 10; i++) dlogits[i] = cache->probabilities[i] - (i == cache->label ? 1.f : 0.f);
    memset(grads->output_weight, 0, sizeof(grads->output_weight));
    memset(grads->output_bias, 0, sizeof(grads->output_bias));
    for (int i = 0; i < 10; i++) {
        grads->output_bias[i] += dlogits[i];
        for (int j = 0; j < 84; j++) {
            grads->output_weight[i][j] += dlogits[i] * cache->f6_tanh[j];
        }
    }
    for (int j = 0; j < 84; j++) {
        float s = 0.f;
        for (int i = 0; i < 10; i++) s += params->output_weight[i][j] * dlogits[i];
        df6_tanh[j] = s;
    }
}

void f6_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, const float df6_tanh[84], float dc5_tanh[120]) {
    float df6_preact[84];
    for (int i = 0; i < 84; i++) {
        df6_preact[i] = df6_tanh[i] * tanh_deriv(cache->f6_preact[i]);
    }
    memset(grads->f6_weight, 0, sizeof(grads->f6_weight));
    memset(grads->f6_bias, 0, sizeof(grads->f6_bias));
    for (int i = 0; i < 84; i++) {
        grads->f6_bias[i] += df6_preact[i];
        for (int j = 0; j < 120; j++) {
            grads->f6_weight[i][j] += df6_preact[i] * cache->c5_tanh[j];
        }
    }
    for (int j = 0; j < 120; j++) {
        float s = 0.f;
        for (int i = 0; i < 84; i++) s += params->f6_weight[i][j] * df6_preact[i];
        dc5_tanh[j] = s;
    }
}

void c5_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, const float dc5_tanh[120], float ds4[16][5][5]) {
    float dc5_preact[120];
    for (int k = 0; k < 120; k++) {
        dc5_preact[k] = dc5_tanh[k] * tanh_deriv(cache->c5_preact[k][0][0]);
    }
    memset(grads->c5_weight, 0, sizeof(grads->c5_weight));
    memset(grads->c5_bias, 0, sizeof(grads->c5_bias));
    for (int k = 0; k < 120; k++) {
        grads->c5_bias[k] += dc5_preact[k];
        for (int c = 0; c < 16; c++) {
            for (int u = 0; u < 5; u++) {
                for (int v = 0; v < 5; v++) {
                    grads->c5_weight[k][c][u][v] += dc5_preact[k] * cache->s4_out[c][u][v];
                }
            }
        }
    }
    for (int c = 0; c < 16; c++) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float s = 0.f;
                for (int k = 0; k < 120; k++) {
                    s += params->c5_weight[k][c][i][j] * dc5_preact[k];
                }
                ds4[c][i][j] = s;
            }
        }
    }
}

void s4_avgpool_backward(const float ds4[16][5][5], float dc3_tanh[16][10][10]) {
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float g = ds4[k][i][j] / 4.f;
                for (int u = 0; u < 2; u++) {
                    for (int v = 0; v < 2; v++) {
                        dc3_tanh[k][i * 2 + u][j * 2 + v] = g;
                    }
                }
            }
        }
    }
}

void c3_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, const float dc3_tanh[16][10][10], float ds2[6][14][14]) {
    float dc3_preact[16][10][10];
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                dc3_preact[k][i][j] = dc3_tanh[k][i][j] * tanh_deriv(cache->c3_preact[k][i][j]);
            }
        }
    }
    memset(grads->c3_weight, 0, sizeof(grads->c3_weight));
    memset(grads->c3_bias, 0, sizeof(grads->c3_bias));
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                float g = dc3_preact[k][i][j];
                grads->c3_bias[k] += g;
                for (int c = 0; c < 6; c++) {
                    for (int u = 0; u < 5; u++) {
                        for (int v = 0; v < 5; v++) {
                            grads->c3_weight[k][c][u][v] += g * cache->s2_out[c][i + u][j + v];
                        }
                    }
                }
            }
        }
    }
    for (int c = 0; c < 6; c++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                float s = 0.f;
                for (int k = 0; k < 16; k++) {
                    for (int u = 0; u < 5; u++) {
                        for (int v = 0; v < 5; v++) {
                            int ip = i - u, jp = j - v;
                            if (ip >= 0 && ip < 10 && jp >= 0 && jp < 10) {
                                s += params->c3_weight[k][c][u][v] * dc3_preact[k][ip][jp];
                            }
                        }
                    }
                }
                ds2[c][i][j] = s;
            }
        }
    }
}

void s2_avgpool_backward(const float ds2[6][14][14], float dc1_tanh[6][28][28]) {
    for (int k = 0; k < 6; k++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                float g = ds2[k][i][j] / 4.f;
                for (int u = 0; u < 2; u++) {
                    for (int v = 0; v < 2; v++) {
                        dc1_tanh[k][i * 2 + u][j * 2 + v] = g;
                    }
                }
            }
        }
    }
}

void c1_backward(const ForwardCache *cache, const Parameters *params, Gradients *grads, const float dc1_tanh[6][28][28]) {
    float dc1_preact[6][28][28];
    for (int k = 0; k < 6; k++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                dc1_preact[k][i][j] = dc1_tanh[k][i][j] * tanh_deriv(cache->c1_preact[k][i][j]);
            }
        }
    }
    memset(grads->c1_weight, 0, sizeof(grads->c1_weight));
    memset(grads->c1_bias, 0, sizeof(grads->c1_bias));
    for (int k = 0; k < 6; k++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                float g = dc1_preact[k][i][j];
                grads->c1_bias[k] += g;
                for (int u = 0; u < 5; u++) {
                    for (int v = 0; v < 5; v++) {
                        grads->c1_weight[k][u][v] += g * cache->input_image[i + u][j + v];
                    }
                }
            }
        }
    }
}
void zero_grads(Gradients *gr) {
    memset(gr, 0, sizeof(*gr));
}
// ---------------- Backward Pass Orchestration ----------------
void backward_pass(const Parameters *params, const ForwardCache *cache, Gradients *grads) {
    zero_grads(grads);

    float df6_tanh[84];
    output_backward(cache, params, grads, df6_tanh);

    float dc5_tanh[120];
    f6_backward(cache, params, grads, df6_tanh, dc5_tanh);

    float ds4[16][5][5];
    c5_backward(cache, params, grads, dc5_tanh, ds4);

    float dc3_tanh[16][10][10];
    s4_avgpool_backward(ds4, dc3_tanh);

    float ds2[6][14][14];
    c3_backward(cache, params, grads, dc3_tanh, ds2);

    float dc1_tanh[6][28][28];
    s2_avgpool_backward(ds2, dc1_tanh);

    c1_backward(cache, params, grads, dc1_tanh);
}

// ---------------- Helpers: Zero Grads, SGD Update, Init Params ----------------

void sgd_update(Parameters *params, const Gradients *grads, float learning_rate) {
    for (int k = 0; k < 6; k++) {
        for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c1_weight[k][u][v] -= learning_rate * grads->c1_weight[k][u][v];
        params->c1_bias[k] -= learning_rate * grads->c1_bias[k];
    }
    for (int k = 0; k < 16; k++) {
        for (int c = 0; c < 6; c++) for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c3_weight[k][c][u][v] -= learning_rate * grads->c3_weight[k][c][u][v];
        params->c3_bias[k] -= learning_rate * grads->c3_bias[k];
    }
    for (int k = 0; k < 120; k++) {
        for (int c = 0; c < 16; c++) for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c5_weight[k][c][u][v] -= learning_rate * grads->c5_weight[k][c][u][v];
        params->c5_bias[k] -= learning_rate * grads->c5_bias[k];
    }
    for (int i = 0; i < 84; i++) {
        for (int j = 0; j < 120; j++) params->f6_weight[i][j] -= learning_rate * grads->f6_weight[i][j];
        params->f6_bias[i] -= learning_rate * grads->f6_bias[i];
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 84; j++) params->output_weight[i][j] -= learning_rate * grads->output_weight[i][j];
        params->output_bias[i] -= learning_rate * grads->output_bias[i];
    }
}

void init_parameters(Parameters *params) {
    srand(42);
    float s1 = 0.1f, s2 = 0.1f, s3 = 0.1f, s4 = 0.1f, s5 = 0.1f;
    for (int k = 0; k < 6; k++) {
        for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c1_weight[k][u][v] = frand_init(s1);
        params->c1_bias[k] = 0.f;
    }
    for (int k = 0; k < 16; k++) {
        for (int c = 0; c < 6; c++) for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c3_weight[k][c][u][v] = frand_init(s2);
        params->c3_bias[k] = 0.f;
    }
    for (int k = 0; k < 120; k++) {
        for (int c = 0; c < 16; c++) for (int u = 0; u < 5; u++) for (int v = 0; v < 5; v++) params->c5_weight[k][c][u][v] = frand_init(s3);
        params->c5_bias[k] = 0.f;
    }
    for (int i = 0; i < 84; i++) {
        for (int j = 0; j < 120; j++) params->f6_weight[i][j] = frand_init(s4);
        params->f6_bias[i] = 0.f;
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 84; j++) params->output_weight[i][j] = frand_init(s5);
        params->output_bias[i] = 0.f;
    }
}

// ---------------- MNIST Loader with Padding ----------------
static uint32_t read_be_u32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { perror("read_u32"); exit(1); }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | ((uint32_t)b[3]);
}

static void load_mnist_images(const char *path, float (*imgs)[32][32], int expected_count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint32_t magic = read_be_u32(f);
    uint32_t n = read_be_u32(f);
    uint32_t h = read_be_u32(f);
    uint32_t w = read_be_u32(f);
    if (magic != 2051 || h != 28 || w != 28) {
        fprintf(stderr, "MNIST images header mismatch: magic=%u, n=%u, h=%u, w=%u\n", magic, n, h, w);
        exit(1);
    }
    if (expected_count > 0 && (int)n != expected_count) {
        fprintf(stderr, "Expected %d images, but file has %u\n", expected_count, n);
        exit(1);
    }
    unsigned char *buf = (unsigned char*)malloc(28 * 28);
    if (!buf) { perror("malloc img buf"); exit(1); }
    for (uint32_t i = 0; i < n; i++) {
        if (fread(buf, 1, 28 * 28, f) != 28 * 28) { fprintf(stderr, "Short read img %u\n", i); exit(1); }
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 32; c++) {
                if (r < 2 || r >= 30 || c < 2 || c >= 30) {
                    imgs[i][r][c] = 0.f; // Padding with zeros
                } else {
                    imgs[i][r][c] = (float)buf[(r - 2) * 28 + (c - 2)] / 255.0f; // Normalize to [0,1]
                }
            }
        }
    }
    free(buf);
    fclose(f);
}

static void load_mnist_labels(const char *path, unsigned char *labels, int expected_count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint32_t magic = read_be_u32(f);
    uint32_t n = read_be_u32(f);
    if (magic != 2049) {
        fprintf(stderr, "MNIST labels header mismatch: magic=%u, n=%u\n", magic, n);
        exit(1);
    }
    if (expected_count > 0 && (int)n != expected_count) {
        fprintf(stderr, "Expected %d labels, but file has %u\n", expected_count, n);
        exit(1);
    }
    if (fread(labels, 1, n, f) != n) { fprintf(stderr, "Short read labels\n"); exit(1); }
    fclose(f);
}

// ---------------- Main Training Loop ----------------
#ifndef TRAIN_IMAGE
#define TRAIN_IMAGE "E:/CNN/train-images.idx3-ubyte"
#define TRAIN_LABEL "E:/CNN/train-labels.idx1-ubyte"
#define TEST_IMAGE  "E:/CNN/t10k-images.idx3-ubyte"
#define TEST_LABEL  "E:/CNN/t10k-labels.idx1-ubyte"
#endif

int main() {
    static float train_images[60000][32][32];
    static unsigned char train_labels[60000];
    static float test_images[10000][32][32];
    static unsigned char test_labels[10000];

    printf("Loading MNIST from %s, %s, %s, %s ...\n", TRAIN_IMAGE, TRAIN_LABEL, TEST_IMAGE, TEST_LABEL);
    load_mnist_images(TRAIN_IMAGE, train_images, 60000);
    load_mnist_labels(TRAIN_LABEL, train_labels, 60000);
    load_mnist_images(TEST_IMAGE, test_images, 10000);
    load_mnist_labels(TEST_LABEL, test_labels, 10000);
    printf("Loaded.\n");

    Parameters params;
    init_parameters(&params);
    ForwardCache cache;
    Gradients grads;

    int epochs = 5;
    float learning_rate = 0.01f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0.f;
        int correct = 0;
        for (int i = 0; i < 60000; i++) {
            int label = (int)train_labels[i];
            loss += forward_pass(train_images[i], label, &params, &cache);

            int predict = 0;
            for (int j = 1; j < 10; j++) if (cache.probabilities[j] > cache.probabilities[predict]) predict = j;
            if (predict == label) correct++;

            backward_pass(&params, &cache, &grads);
            sgd_update(&params, &grads, learning_rate);
        }
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%%\n", epoch + 1, loss / 60000.0f, 100.0f * (float)correct / 60000.0f);

        float test_loss = 0.f;
        int test_correct = 0;
        for (int i = 0; i < 10000; i++) {
            int label = (int)test_labels[i];
            test_loss += forward_pass(test_images[i], label, &params, &cache);
            int predict = 0;
            for (int j = 1; j < 10; j++) if (cache.probabilities[j] > cache.probabilities[predict]) predict = j;
            if (predict == label) test_correct++;
        }
        printf("  -> Test  - Loss: %.4f - Accuracy: %.2f%%\n", test_loss / 10000.0f, 100.0f * (float)test_correct / 10000.0f);
    }

    return 0;
}