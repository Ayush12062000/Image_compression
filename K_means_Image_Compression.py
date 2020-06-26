#IMPORTING LIBRARIES

#%%
from __future__ import print_function
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use("ggplot")

from skimage import io
from sklearn.cluster import KMeans

from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets

# %%
plt.rcParams['figure.figsize']= (20,12)

#%%
os.chdir('C:/Users/Ayush/Desktop/Images_compression/Images')
os.listdir()

# %%
img = io.imread('C:/Users/Ayush/Desktop/Images_compression/Images/baby-groot-5k-2018-artwork_1539978750.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(img)

# %%
img.shape

# %%
img_data = (img / 255.0).reshape(-1, 3)
img_data.shape

# %%
from plot_utils import plot_utils
x = plot_utils(img_data, title = "Input color space: over 16 million possible colors")
x.colorSpace()

# %%
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16).fit(img_data)
k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]

y = plot_utils(img_data, colors = k_colors, title = "Reduced color space:16 colors")
y.colorSpace()

# %%
img_dir = 'C:/Users/Ayush/Desktop/Images_compression/Images/'

# %%
@interact
def color_compression(image = os.listdir(img_dir), k = IntSlider(min=1, max=256,step=1,value=16,continuous_update=False,layout=dict(width='100%'))):

    input_img = io.imread(img_dir+image)
    img_data = (input_img / 255.0).reshape(-1, 3)

    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]

    k_img = np.reshape(k_colors, (input_img.shape))

    fig , (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle('K-Means Image Compression', fontsize=20)

    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)

    ax2.set_title('Original (16,777,216 colors)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(input_img)

    plt.subplots_adjust(top=0.85)
    plt.show()


# %%
