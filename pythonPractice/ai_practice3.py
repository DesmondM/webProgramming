import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, uniform_filter

feature_map = np.array([
    [2,4,7,0],
    [8,9,3,1],
    [5,6,1,2],
    [2,3,4,3],
])

max_pooled = maximum_filter(feature_map, size=2, mode='constant')
avg_pooled = uniform_filter(feature_map, size=2, mode='constant')

fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(feature_map, cmap='viridis')
axes[0].set_title('Original Feature map')
axes[1].imshow(max_pooled, cmap='viridis')
axes[1].set_title('Max Pooled')
axes[2].imshow(avg_pooled, cmap='viridis')
axes[2].set_title('Average pooled')
plt.show()