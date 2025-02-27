







import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color


def read_image(path):

    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb
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
    """Quantizes image by averaging the colors in each bucket."""
    r_average = np.mean(img_arr[:, 0])
    g_average = np.mean(img_arr[:, 1])
    b_average = np.mean(img_arr[:, 2])
    
    for data in img_arr:
        img[data[3], data[4]] = [r_average, g_average, b_average]

def split_into_buckets(img, img_arr, depth):
    """Recursively splits the color space into buckets and quantizes the image."""
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
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

def to_hsl(image):
    """Convert a BGR image to HSL."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)

def to_rgb(image, color_space):
    """Convert an image from HSV or HLS back to RGB."""
    if color_space == "HSV":
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    elif color_space == "HLS":
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HLS2RGB)
    else:
        raise ValueError("Invalid color space. Use 'HSV' or 'HLS'.")

def adjust_hue(image, hue_shift=0):
    """Adjust Hue without modifying other properties."""
    hsv = to_hsv(image)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180  # Keep within [0, 179]
    return to_rgb(hsv, "HSV")

def adjust_saturation(image, saturation_scale=1.0):
    """Adjust Saturation without modifying other properties."""
    hsv = to_hsv(image)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)  # Scale within range
    return to_rgb(hsv, "HSV")

def adjust_brightness(image, brightness_shift=0):
    """Adjust Brightness (Value in HSV) without affecting other properties."""
    hsv = to_hsv(image)
    hsv[..., 2] = np.clip(hsv[..., 2] + brightness_shift, 0, 255)  # Shift within range
    return to_rgb(hsv, "HSV")

def adjust_lightness(image, lightness_shift=0):
    """Adjust Lightness (L in HLS) without modifying other properties."""
    hls = to_hsl(image)
    hls[..., 1] = np.clip(hls[..., 1] + lightness_shift, 0, 255)  # Shift within range
    return to_rgb(hls, "HLS")


# =======================================
                                # Part 4
# =======================================


file_name = "images/image2.jpg"
import numpy as np
from skimage import color

def deltaE(sampleRGB1, sampleRGB2):
    """Calculate DeltaE (CIEDE2000) color difference between two RGB colors."""
    # Normalize the RGB values to [0, 1] range
    sampleRGB1 = np.array(sampleRGB1, dtype=np.float32) / 255.0
    sampleRGB2 = np.array(sampleRGB2, dtype=np.float32) / 255.0

    # Convert RGB to LAB color space
    lab1 = color.rgb2lab(sampleRGB1.reshape(1, 1, 3))[0, 0]
    lab2 = color.rgb2lab(sampleRGB2.reshape(1, 1, 3))[0, 0]

    # Calculate DeltaE (CIEDE2000)
    dE = color.deltaE_ciede2000(lab1, lab2)

    # Categorize the DeltaE value
    if dE <= 1.0:
        category = "Not perceptible by human eyes"
    elif 1.0 < dE <= 2.0:
        category = "Perceptible through close observation"
    elif 2.0 < dE <= 10.0:
        category = "Perceptible at a glance"
    elif 10.0 < dE <= 49.0:
        category = "Colors are more similar than opposite"
    else:
        category = "Colors are exact opposite"

    # Print the DeltaE and category
    print(f"DeltaE: {dE:.2f} | Category: {category}")
    return dE

def resize_image(image, max_size=800):

    
    if image is None:
        print("Error: Could not read the image!")
        return None

    height, width = image.shape[:2]

    # If either dimension exceeds max_size
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)  # Scale based on the largest dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image