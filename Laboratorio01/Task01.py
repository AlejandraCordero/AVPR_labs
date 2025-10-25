# Task 1: Edge Detection Comparison
# Full name: Maria Alejandra Mamani Cordero

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image_exercise.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise Exception("Image not found. Please check the file path.")

#
# Sobel Operator
sobel_x = cv2.Sobel(image, -1, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, -1, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

# Prewitt
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
prewitt_x_edges = cv2.filter2D(image, -1, prewitt_x)
prewitt_y_edges = cv2.filter2D(image, -1, prewitt_y)
prewitt_magnitude = np.sqrt(prewitt_x_edges**2 + prewitt_y_edges**2)
prewitt_magnitude = np.uint8(np.clip(prewitt_magnitude, 0, 255))

# Scharr Operator
scharr_x = cv2.Scharr(image, -1, 1, 0)
scharr_y = cv2.Scharr(image, -1, 0, 1)
scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
scharr_magnitude = np.uint8(np.clip(scharr_magnitude, 0, 255))

# Results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(sobel_magnitude, cmap='gray')
axes[0, 1].set_title('Sobel Edge Detection')
axes[0, 1].axis('off')

axes[1, 0].imshow(prewitt_magnitude, cmap='gray')
axes[1, 0].set_title('Prewitt Edge Detection')
axes[1, 0].axis('off')

axes[1, 1].imshow(scharr_magnitude, cmap='gray')
axes[1, 1].set_title('Scharr Edge Detection')
axes[1, 1].axis('off')

plt.tight_layout(pad=3.0)
plt.show()
