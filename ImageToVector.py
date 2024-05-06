import cv2

# Read the image
image = cv2.imread("./Hiragana/ka.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Flatten the grayscale image into a 1-D array
flat_array = gray_image.flatten()

print(list(flat_array))
