import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from utils import save_image



image_path = 'images/image1.jpeg'

#if you want images to save 
save = True
image_output_name = 'result.jpeg'



script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = os.path.join(script_dir, image_path)

image = cv2.imread(file_name)  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to HLS for easier manipulation
hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)

# Display the initial image
img_display = ax.imshow(image)
ax.axis("off")

# Define sliders
ax_hue = fig.add_axes([0.25, 0.25, 0.65, 0.03])
hue_slider = Slider(ax=ax_hue, label="Hue", valmin=-179, valmax=179, valinit=0)

ax_saturation = fig.add_axes([0.25, 0.2, 0.65, 0.03])
sat_slider = Slider(ax=ax_saturation, label="Saturation", valmin=0.1, valmax=5, valinit=1)

ax_brightness = fig.add_axes([0.25, 0.15, 0.65, 0.03])
bright_slider = Slider(ax=ax_brightness, label="Brightness", valmin=0.1, valmax=5, valinit=1)

ax_lightness = fig.add_axes([0.25, 0.1, 0.65, 0.03])
light_slider = Slider(ax=ax_lightness, label="Lightness", valmin=0.1, valmax=5, valinit=1)


# Update function
def update(val):
    h = hue_slider.val
    s = sat_slider.val
    v = bright_slider.val
    l = light_slider.val
    
    # Modify HLS values
    new_hls = hls_image.copy()
    new_hls[:, :, 0] = (new_hls[:, :, 0] + h) % 180  # Hue wrapping
    new_hls[:, :, 1] = np.clip(new_hls[:, :, 1] * l, 0, 255)  # Lightness scaling
    new_hls[:, :, 2] = np.clip(new_hls[:, :, 2] * s, 0, 255)  # Saturation scaling
    
    # Convert back to RGB
    new_rgb = cv2.cvtColor(new_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)
    new_rgb = np.clip(new_rgb * v, 0, 255).astype(np.uint8)  # Apply brightness after conversion
    
    img_display.set_data(new_rgb)
    fig.canvas.draw_idle()

# Connect sliders to update function
hue_slider.on_changed(update)
sat_slider.on_changed(update)
bright_slider.on_changed(update)
light_slider.on_changed(update)

# Reset 
reset_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset")

def reset(event):
    hue_slider.reset()
    sat_slider.reset()
    bright_slider.reset()
    light_slider.reset()

reset_button.on_clicked(reset)

def on_close(event):
    if save:
        save_image(img_display.get_array(), script_dir, image_output_name,2)

fig.canvas.mpl_connect('close_event', on_close)

plt.show()
