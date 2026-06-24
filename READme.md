# **SegFormer**

# **Overview**

This repository provides a complete implementation of the SegFormer-B0 semantic segmentation model in C++ for edge devices.

The model is implemented entirely from scratch without relying on deep learning frameworks such as PyTorch, TensorFlow, ONNX Runtime, or OpenVINO during inference.

The repository contains two implementations:

1. **Naive Implementation** – Pure C++ implementation without external linear algebra libraries.  
2. **OpenBLAS Implementation** – Optimized implementation using OpenBLAS for faster inference.

### **Features**

* Pure C++ inference  
* Complete implementation of SegFormer-B0 architecture  
* Naive and optimized implementations  
* OpenBLAS accelerated matrix operations  
* OpenMP based parallelization  
* Deployment on resource-constrained edge devices such as Raspberry Pi 4

---

## **Dependencies**

### **Naive Implementation**

* C++14 or higher  
* OpenCV (\>= 4.x)  
* CMake (\>= 3.10)

### **OpenBLAS Implementation**

* C++14 or higher  
* OpenCV (\>= 4.x)  
* OpenBLAS  
* OpenMP  
* CMake (\>= 3.10)

---

## **Installing Dependencies**

## **Ubuntu / Raspberry Pi OS**

  \-  sudo apt update  
  \-  sudo apt install cmake g++ libopencv-dev libopenblas-dev libomp-dev

### **Windows**

Install the following:

* OpenCV  
* OpenBLAS (required only for optimized implementation)  
* CMake  
* MinGW/MSVC compiler

Update the paths inside `CMakeLists.txt` accordingly.

---

## **Building the Project**

### **Clone Repository**

git clone https://github.com/MeenaakshiAtkade/SegFormer.git

cd SegFormer

---

### **Build Naive Implementation**

cd Naive\_Implementation

mkdir build  
cd build

cmake ..  
make \-j4

Run:

./segformer “input image path”

---

### **Build OpenBLAS Implementation**

cd OpenBLAS\_Implementation

mkdir build  
cd build

cmake ..  
make \-j4

Run:

./segformer “input image path”

---

## **Pretrained Weights**

This implementation uses pretrained SegFormer-B0 weights fine-tuned on the Cityscapes dataset.

Download the pretrained weights and place all `.bin` files inside:

PreTrainedWeights/

Inside `main.cpp`, specify:

string weights\_dir \= "/path/to/PreTrainedWeights";

---

## **Experimental Platform**

The implementations have been tested using:

* Raspberry Pi 4  
* Desktop CPU  
* USB Webcam  
* Headphones  
* Power Bank

---

