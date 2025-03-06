import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color
import os

def read_image(path):

    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb



def resize_image(image, max_size=800):

    
    if image is None:
        print("Error: Could not read the image!")
        return None

    height, width = image.shape[:2]

    # If either dimension exceeds max_size
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)  # pick largest dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

import cv2
import os

# Function to save the image
import cv2
import os

# Function to save the image
def save_image(image, script_dir, output_name, task_index):
    # Create the "outputs" directory inside script_dir if it doesn't exist
    outputs_folder = os.path.join(script_dir, 'outputs')
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
    
    # Create the folder "output{task_index}" inside the "outputs" folder
    output_folder = os.path.join(outputs_folder, f'output{task_index}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file path with the given output_name
    output_path = os.path.join(output_folder, output_name)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")

# =======================================
                                # Part 1
# =======================================

def grayscale_average(image):
    return np.mean(image, axis=2).astype(np.uint8)

def grayscale_weighted(image):
    im = 0.299* image[:, :, 2]+ 0.587 *image[:, :, 1]+ 0.114 * image[:, :, 0] 
    return im.astype(np.uint8)
def mse(image1, image2):
    assert image1.shape==image2.shape
    return np.sum((image1-image2)**2/float(image1.shape[0] * image1.shape[1]))

# =======================================
                                # Part 2
# =======================================

def kmeans_quantization(original_image, k):
    reshaped_image = np.reshape(original_image, 
      ((original_image.shape[0] * original_image.shape[1]), 3))
    model = KMeans(n_clusters=k)
    target = model.fit_predict(reshaped_image)
    color_space = model.cluster_centers_
    output_image = np.reshape(color_space.astype(
    "uint8")[target], (original_image.shape[0], original_image.shape[1], 3))
    return output_image
def uniform_quantization(image, k, colour_range = 256):
    output = (image // (colour_range/k))*(colour_range/k)
    output = np.clip(output, 0, colour_range - 1)
    return output.astype(np.uint8)

def median_cut_quantize(img, img_arr):
    r_average = np.mean(img_arr[:, 0])
    g_average = np.mean(img_arr[:, 1])
    b_average = np.mean(img_arr[:, 2])
    
    for data in img_arr:
        img[data[3], data[4]] = [r_average, g_average, b_average]

def split_into_buckets(img, img_arr, depth):
    if len(img_arr) == 0:
        return 
    
    if depth == 0:
        median_cut_quantize(img, img_arr)
        return
    
    r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])
    
    space_with_highest_range = 0
    if g_range >= r_range and g_range >= b_range:
        space_with_highest_range = 1
    elif b_range >= r_range and b_range >= g_range:
        space_with_highest_range = 2
    elif r_range >= b_range and r_range >= g_range:
        space_with_highest_range = 0

    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)

    split_into_buckets(img, img_arr[:median_index], depth - 1)
    split_into_buckets(img, img_arr[median_index:], depth - 1)

def median_quantization(img, depth):
    flattened_img_array = []
    for rindex, rows in enumerate(img):
        for cindex, color in enumerate(rows):
            flattened_img_array.append([color[0], color[1], color[2], rindex, cindex])
    
    flattened_img_array = np.array(flattened_img_array)

    split_into_buckets(img, flattened_img_array, depth)
    

    return img




# =======================================
                                # Part 3
# =======================================


def to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

def to_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)

def to_rgb(image, color_space):
    if color_space == "HSV":
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    elif color_space == "HLS":
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HLS2RGB)
    else:
        raise ValueError("Invalid color space. Use 'HSV' or 'HLS'.")

def adjust_hue(image, hue_shift=0):
    hsv = to_hsv(image)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180  # Keep within [0, 179]
    return to_rgb(hsv, "HSV")

def adjust_saturation(image, saturation_scale=1.0):
    hsv = to_hsv(image)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255) 
    return to_rgb(hsv, "HSV")

def adjust_brightness(image, brightness_shift=0):
    hsv = to_hsv(image)
    hsv[..., 2] = np.clip(hsv[..., 2] + brightness_shift, 0, 255)  
    return to_rgb(hsv, "HSV")

def adjust_lightness(image, lightness_shift=0):
    hls = to_hsl(image)
    hls[..., 1] = np.clip(hls[..., 1] + lightness_shift, 0, 255)  
    return to_rgb(hls, "HLS")
