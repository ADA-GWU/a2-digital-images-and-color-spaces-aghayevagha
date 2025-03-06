import cv2
import numpy as np
import os
from skimage.color import deltaE_ciede2000, rgb2lab
from utils import resize_image, save_image

# Image path setup
image_path = 'images/image1.jpeg'
#if you want images to save 
save = True
image_output_name = 'result.jpeg'


script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = os.path.join(script_dir, image_path)

# Global variables
image = None
threshold = 20  # Set threshold for color similarity
label_colour = (0, 255, 0)  # Replace similar colors with label colour
modified_image = None  # Store modified image for saving


def on_mouse_click(event, x, y, flags, param):
    global image, lab_image, modified_image

    if event == cv2.EVENT_LBUTTONDOWN:
 
        selected_color_bgr = image[y, x].astype(np.float32) / 255.0
        selected_color_lab = rgb2lab([[selected_color_bgr]])

        delta_e_map = deltaE_ciede2000(lab_image, selected_color_lab)

        mask = delta_e_map < threshold

        modified_image = image.copy()
        modified_image[mask] = label_colour

        # Show updated image
        cv2.imshow("Modified Image", modified_image)

        if save:
            save_image(modified_image, script_dir, image_output_name, 4) 
            print("Image saved.")


def main():
    global image, lab_image, modified_image

    # Load image
    image = cv2.imread(file_name)  # Uses the correct file path
    image = resize_image(image, 800)
    if image is None:
        print(f"Error loading image: {file_name}")
        return
    

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0 
    lab_image = rgb2lab(image_rgb)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", on_mouse_click)

    # Wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
