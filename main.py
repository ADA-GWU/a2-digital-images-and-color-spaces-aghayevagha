import os
import cv2
import numpy as np
from skimage import color
from utils import deltaE, resize_image



# make sure the images folder and main script is in the same directory 
# the first image contains different colours, good for test
image_path = 'images/image1.jpeg'


points = []
image = None
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = os.path.join(script_dir, image_path )




def mouse_callback(event, x, y, flags, param):
    global points, image

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Store clicked points
        print(f"Point {len(points)} selected at ({x}, {y})")

        if len(points) == 2:
            # Get RGB values from image
            color1 = image[y, x]  # BGR format
            color2 = image[points[0][1], points[0][0]]  # First point color
            color1_rgb = color1[::-1]
            color2_rgb = color2[::-1]

            dE = deltaE(color1_rgb, color2_rgb)
            points = []

def main():
    global image

    image = cv2.imread(filename=file_name)
    if image is None:
        print("no image!")
        return
    image= resize_image(image, 800)
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

