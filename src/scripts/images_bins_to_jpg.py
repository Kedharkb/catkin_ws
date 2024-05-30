import os

import cv2
import numpy as np


def create_image_from_binary(bin_file_path, img_file_path,desired_encoding="bgr8"):
    # Read binary data from file
    with open(bin_file_path, 'rb') as f:
        binary_data = f.read()

    # Decode binary data to image array
    # Assuming the data is stored as raw pixel values
    image_array = np.frombuffer(binary_data, dtype=np.uint8)
    print(image_array.size)
        # Reshape image array to the desired shape (height, width, channels)
    height, width, channels = 1080, 1920, 3  # Image size provided by user
    image_array = image_array.reshape((height, width, channels))

    # Convert the color encoding if needed
    if desired_encoding.lower() != "bgr8":
        image_array = cv2.cvtColor(image_array, getattr(cv2, f'COLOR_BGR2{desired_encoding.upper()}'))

    # Save the image using OpenCV
    cv2.imwrite(img_file_path, image_array)

    # Reshape image array to the desired shape (height, width, channels)
    # Make sure you know the dimensions and color channels of the image
    # For example, if it's a grayscale image with dimensions (height, width):
    # image_array = image_array.reshape((height, width))
    # For a color image with dimensions (height, width, channels):
    # image_array = image_array.reshape((height, width, channels))



if __name__=='__main__':
    camera_bin_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/camera_images_bins'
    camera_images_dir = '/home/kedhar/workspace/catkin_ws/src/scripts/camera_images'
    camera_bins = sorted(os.listdir(camera_bin_dir))
    for idx, (camera_file) in enumerate(camera_bins):
        camera_bin_path= os.path.join(camera_bin_dir, camera_file)
        create_image_from_binary(camera_bin_path, f"{camera_images_dir}/camera_{idx}.jpg")