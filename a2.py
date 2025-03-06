import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.cluster import KMeans
from utils import resize_image, uniform_quantization, kmeans_quantization, median_quantization


# make sure the images folder and main script is in the same directory 
# the first image contains different colours, good for test
image_path = 'images/image5.jpeg'

# if you want reduce computation, resize the image and set it to True, give a maximum image 
resize = True
max_size = 300




script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = os.path.join(script_dir, image_path)
image = cv2.imread(file_name)
if resize:
    image = resize_image(image,max_size) # resize image to achiave faster computation
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original_image = image.copy()

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.5)
img_display = ax.imshow(image)
ax.axis("off")

# Define sliders for each method
ax_uniform = fig.add_axes([0.25, 0.3, 0.65, 0.03])
slider_uniform = Slider(ax=ax_uniform, label="Uniform", valmin=2, valmax=32, valinit=8, valstep=1)

ax_median = fig.add_axes([0.25, 0.25, 0.65, 0.03])
slider_median = Slider(ax=ax_median, label="Median Cut", valmin=1, valmax=30, valinit=6, valstep=1)
ax_median.set_visible(False)

ax_kmeans = fig.add_axes([0.25, 0.2, 0.65, 0.03])
slider_kmeans = Slider(ax=ax_kmeans, label="K-Means", valmin=2, valmax=30, valinit=6, valstep=1)
ax_kmeans.set_visible(False)

# Radio buttons
ax_radio = fig.add_axes([0.05, 0.4, 0.15, 0.15])
radio = RadioButtons(ax_radio, ["Uniform", "Median Cut", "K-Means"])

selected_method = "Uniform"

def apply_quantization():
    global selected_method
    if selected_method == "Uniform":
        k = int(slider_uniform.val)
        img_display.set_data(uniform_quantization(original_image.copy(), k))
    elif selected_method == "Median Cut":
        depth = int(slider_median.val)
        img_display.set_data(median_quantization(original_image.copy(), depth))
    elif selected_method == "K-Means":
        k = int(slider_kmeans.val)
        img_display.set_data(kmeans_quantization(original_image.copy(), k))
    fig.canvas.draw_idle()

def update_uniform(val):
    if selected_method == "Uniform":
        apply_quantization()

def update_median(val):
    if selected_method == "Median Cut":
        apply_quantization()

def update_kmeans(val):
    if selected_method == "K-Means":
        apply_quantization()

def select_method(label):
    global selected_method
    selected_method = label
    
    slider_uniform.set_val(8)
    slider_median.set_val(6)
    slider_kmeans.set_val(6)
    
    ax_uniform.set_visible(label == "Uniform")
    ax_median.set_visible(label == "Median Cut")
    ax_kmeans.set_visible(label == "K-Means")
    
    apply_quantization()

def reset(event):
    global selected_method
    selected_method = "Uniform"
    radio.set_active(0)
    select_method("Uniform")
    fig.canvas.draw_idle()

radio.on_clicked(select_method)
slider_uniform.on_changed(update_uniform)
slider_median.on_changed(update_median)
slider_kmeans.on_changed(update_kmeans)

reset_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset")
reset_button.on_clicked(reset)

# Apply initial quantization based on default method
apply_quantization()

plt.show()
