# BME1312

## Course Introduction

Smart medicine, with Artificial Intelligence (AI) as its core technology, is a significant direction for future medical development. Medical imaging represents one of the most important applications of AI in the medical field today. Students in medical imaging need to learn basic AI knowledge and develop the fundamental skills for AI algorithm development. Meanwhile, students with backgrounds in information science and technology need to understand the specific requirements of imaging and medical imaging for AI.

This cross-disciplinary course aims to address the knowledge gaps for students from both backgrounds, enabling them to excel in intelligent medical imaging, conduct innovative research, and contribute to the healthcare industry.

## Project

### Project 1: MRI Image Reconstruction

[code](https://github.com/XiongWenye/Deep-Learning-Dynamic-MRI-Reconstruction)

- **Objective:** Reconstruct high-quality MRI images from undersampled k-space data.
- **Approach:** Implement a U-Net and a 3D residual network.
- **Techniques:** Employ dropout, dynamic learning rate scheduling and data augmentation to optimize performance.
 
Generate variable density undersampling patterns with acceleration factor 5 and central k-space sampling, analyze the resulting aliasing artifacts, and evaluate reconstruction performance with PSNR and SSIM metrics.

## Labs

### Lab 0: Handwritten Digit Recognition

- **Task:** Build a simple LeNet-style CNN to classify handwritten digits.
- **Loss Function:** Cross-entropy loss.

### Lab 1: Advanced MRI Reconstruction

- **Task:** Reconstruct MRI images from undersampled k-space data using a Cascade Network.

#### Enhancements:

- Integrate dropout and data augmentation for improved robustness and accuracy.

### Lab 2

- **Task:** Implement a U-Net-based convolutional neural network for 2D medical image segmentation.
 
 #### Enhancements:

- Data augmentation.
- Segmentation metrics: accuracy, sensitivity, specificity, precision, F1, Jaccard (IoU), and Dice coefficient.
- Use binary cross-entropy loss for training.
- With learning rate scheduling and validate on a separate set.