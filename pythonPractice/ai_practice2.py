import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

#Load a sample greyscale image
image=np.random.rand(10,10)
print(image)

edge_detection_kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1],
])

blur_kernel= np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1],
])/9

edge_detected_image = convolve(image, edge_detection_kernel)
blurred_image=convolve(image, blur_kernel)

fig, axes = plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(edge_detected_image, cmap='gray')
axes[1].set_title("Edge detected")
axes[2].imshow(blurred_image, cmap='gray')
axes[2].set_title("Blurred")
plt.show()