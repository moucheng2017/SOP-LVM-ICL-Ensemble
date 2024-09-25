from PIL import Image
import pytesseract
import cv2
import numpy as np

# Path to the screenshot image file
screenshot_path = 'path_to_your_screenshot.png'

# Open the image using Pillow
img = Image.open(screenshot_path)

# Use pytesseract to extract text
extracted_text = pytesseract.image_to_string(img)

# Print the extracted text
print(extracted_text)

boxes = pytesseract.image_to_boxes(img)
for box in boxes.splitlines():
    b = box.split(' ')
    print(f'Character: {b[0]}, Coordinates: (x1={b[1]}, y1={b[2]}, x2={b[3]}, y2={b[4]})')


# Load the screenshot
img = cv2.imread(screenshot_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use thresholding to detect UI elements
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours of UI elements
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected UI elements
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save or display the result
cv2.imshow('Detected UI Elements', img)
cv2.waitKey(0)
cv2.destroyAllWindows()