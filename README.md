🧠 CNN Implementation in C from Scratch
📌 Introduction
This project implements a Convolutional Neural Network (CNN) from the ground up using the C programming language, without relying on pre-built deep learning libraries like TensorFlow or PyTorch. The primary goals are:

Gain a deep understanding of CNN operations at a low level.
Build the entire pipeline, from forward propagation to classification, manually.
Experiment with the handwritten digit recognition task using the MNIST dataset (28x28 grayscale images, 10 classes).

⚙️ Model Architecture
The implemented CNN follows the architecture depicted below:

The layers are:

Input Layer: 28×28×1 input image.
Convolution Layer 1: 5×5 kernel, stride 1, output 24×24×2 → ReLU activation.
MaxPooling Layer 1: 2×2 kernel → output 12×12×2.
Convolution Layer 2: 3×3 kernel, 4 filters → output 10×10×4 → Sigmoid activation.
MaxPooling Layer 2: 2×2 kernel → output 5×5×4.
Flatten Layer: Converts 5×5×4 tensor to a 100-dimensional vector.
Fully Connected Layer:
100 nodes → 10 nodes.
Activation function: Softmax.


Output: Predicts digits from 0–9.

📂 Directory Structure
├── src/
│   ├── cnn.c          # CNN implementation
│   ├── layers.c       # Layer implementations: conv, pooling, fc
│   ├── utils.c        # Utility functions: data loading, weight initialization
│   └── main.c         # Main program
├── data/
│   └── mnist/         # MNIST dataset (.idx format)
├── include/
│   └── *.h            # Header files
└── README.md

🚀 How to Run

Clone the repository:

git clone https://github.com/<username>/cnn-from-scratch-c.git
cd cnn-from-scratch-c


Compile the code:

gcc src/*.c -o cnn -lm


Execute the program:

./cnn


Expected Output:
Displays loss and accuracy per epoch.
Prints predictions for a few MNIST images.



📊 Training Results
The model was trained on the MNIST dataset with the following results:
Loading MNIST from E:/CNN/train-images.idx3-ubyte, E:/CNN/train-labels.idx1-ubyte, E:/CNN/t10k-images.idx3-ubyte, E:/CNN/t10k-labels.idx1-ubyte ...
Loaded.
Epoch 1 - Loss: 0.3812 - Accuracy: 88.58%
  -> Test - Loss: 0.1728 - Accuracy: 94.72%
Epoch 2 - Loss: 0.1506 - Accuracy: 95.60%
  -> Test - Loss: 0.1400 - Accuracy: 95.62%
Epoch 3 - Loss: 0.1281 - Accuracy: 96.24%
  -> Test - Loss: 0.1253 - Accuracy: 96.06%
Epoch 4 - Loss: 0.1170 - Accuracy: 96.56%
  -> Test - Loss: 0.1242 - Accuracy: 96.18%
Epoch 5 - Loss: 0.1101 - Accuracy: 96.65%
  -> Test - Loss: 0.1170 - Accuracy: 96.37%


Achieves a test accuracy of up to 96.37% after 5 epochs.
Training time is longer compared to Python/Frameworks due to the absence of GPU optimization.

📚 Applied Knowledge

Core concepts: Convolution, Pooling, Flatten, Fully Connected, Softmax.
Forward propagation mechanics.
Loss function: Cross-Entropy.
Optimization technique: Stochastic Gradient Descent (SGD).
