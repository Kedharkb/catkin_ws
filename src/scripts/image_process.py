import cv2

# Read the saved image
image_path = "/home/kedhar/Desktop/left0000.jpg"
saved_image = cv2.imread(image_path)

if saved_image is None:
    print("Error: Could not read the image.")
else:
    # Split the image into its color channels
    b_channel, g_channel, r_channel = cv2.split(saved_image)

    # Display each color channel
    cv2.imshow("Blue Channel", b_channel)
    cv2.imshow("Green Channel", g_channel)
    cv2.imshow("Red Channel", r_channel)

    # Combine the channels to visualize the image
    bgr_image = cv2.merge([b_channel, g_channel, r_channel])
    cv2.imshow("BGR Image", bgr_image)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()