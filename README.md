# CNN triển khai bằng ngôn ngữ C 

## 📌 Giới thiệu
Dự án này triển khai một **Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN)** từ đầu bằng ngôn ngữ **C** mà **không sử dụng thư viện học sâu (deep learning library)**.  
Mục tiêu chính: xây dựng mô hình CNN cơ bản để nhận diện ảnh kích thước **28x28** (ví dụ MNIST - chữ số viết tay).  

---

## 🖼️ Sơ đồ kiến trúc CNN

![CNN Architecture](cnn-architecture.jpeg)

---

## ⚙️ Kiến trúc mô hình
Mạng CNN được thiết kế theo sơ đồ trên với các bước:

1. **Input Layer**  
   - Kích thước: `28x28x1` (ảnh xám)

2. **Convolutional Layer 1**  
   - 2 filter kích thước `5x5`  
   - Kết quả: `24x24x2`  
   - Activation: **ReLU**

3. **MaxPooling Layer 1**  
   - Kích thước kernel: `2x2`  
   - Kết quả: `12x12x2`

4. **Convolutional Layer 2**  
   - 4 filter kích thước `3x3`  
   - Kết quả: `10x10x4`  
   - Activation: **Sigmoid**

5. **MaxPooling Layer 2**  
   - Kích thước kernel: `2x2`  
   - Kết quả: `5x5x4`

6. **Flatten Layer**  
   - Chuyển từ tensor `5x5x4` → vector `100 node`

7. **Fully Connected Layer**  
   - 100 node ẩn → 10 node đầu ra (ứng với 10 lớp số `0-9`)

8. **Output Layer**  
   - Hàm softmax để phân loại.

---

## 🛠️ Công nghệ sử dụng
- Ngôn ngữ lập trình: **C**
- Thư viện chuẩn: `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`, `<time.h>`, `<stdint.h>`

---

## 🚀 Chạy thử dự án
```bash
# Biên dịch
gcc main.c -o cnn -lm

# Chạy chương trình
./cnn
