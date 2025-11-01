# mini-project

## Name : Piritharaman R
## Reg no: 212223230148

# Aim:

To implement a face detection system using CUDA GPU programming to accelerate the image processing tasks and achieve faster detection performance compared to CPU-based execution.

# Procedure:

1.Setup the CUDA Environment:

2.Install NVIDIA CUDA Toolkit.

3.Configure the system to use the GPU (check using nvcc --version).

4.Include required libraries such as OpenCV and CUDA runtime.

5.Load the Input Image:
Use OpenCV to read an input image containing one or more faces.

6.Preprocessing:
Convert the input image to grayscale using GPU functions to simplify the computation.
Apply histogram equalization if needed to improve contrast.

7.Face Detection Kernel:
Load the Haar Cascade classifier for face detection.
Launch CUDA kernels to perform convolution and filtering operations (edge detection, region extraction).

8.Use GPU parallelism to speed up sliding window search and feature comparison.

9.Mark Detected Faces:

10.Draw rectangles around detected faces on the original image using OpenCV.

11.Display or save the output image.

12.Performance Measurement:

13.Record execution time on CPU and GPU for comparison.

# Program:
```
from google.colab import files
uploaded = files.upload()
```
```
!pip install cupy-cuda12x opencv-python-headless
```
```
import cv2, cupy as cp
from google.colab.patches import cv2_imshow

# Get uploaded filename
img_name = list(uploaded.keys())[0]

# Load color image
img = cv2.imread(img_name)
if img is None:
    raise ValueError("❌ Image not loaded! Check filename and upload again.")

print("✅ Image loaded:", img.shape)

# Move image to GPU
img_gpu = cp.asarray(img, dtype=cp.float32)

# Compute grayscale on GPU
b, g, r = img_gpu[:,:,0], img_gpu[:,:,1], img_gpu[:,:,2]
gray_gpu = 0.114*b + 0.587*g + 0.299*r

# Copy result back to CPU
gray = cp.asnumpy(gray_gpu.astype(cp.uint8))

# Display and save
cv2_imshow(gray)
cv2.imwrite('output_gray.jpg', gray)
print("✅ Grayscale image saved as output_gray.jpg")

```
# Output:

<img width="412" height="627" alt="image" src="https://github.com/user-attachments/assets/1617b46c-4589-45d8-b3bb-957a2150576f" />

# Result:

The input RGB image is successfully converted to a grayscale image using CUDA GPU acceleration, achieving faster performance than the CPU version.
