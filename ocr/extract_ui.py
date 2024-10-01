import cv2
import pytesseract
from PIL import Image
import json
import os

def extract_ui_elements(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(Image.fromarray(gray))

    # save the text to a jason file:
    print(text)

    # save the text to a file 
    
    # # Save the entire image as a detected UI element
    # element_image_path = f"{os.path.splitext(image_path)[0]}_element.png"
    # cv2.imwrite(element_image_path, img)
    
    # # Add metadata
    # metadata = [{
    #     'type': 'detected_element',  # This should be determined by a more advanced method
    #     'location': {'x': 0, 'y': 0, 'width': img.shape[1], 'height': img.shape[0]},
    #     'image_path': element_image_path,
    #     'text': text,
    #     'tag': 'unknown',  # This should be determined by a more advanced method
    #     'xpath': 'unknown'  # This should be determined by a more advanced method
    # }]
    
    # # Save metadata to JSON file
    # metadata_file = os.path.splitext(image_path)[0] + '_metadata.json'
    # with open(metadata_file, 'w') as f:
    #     json.dump(metadata, f, indent=4)

# Example usage
image_path = '/home/moucheng/projects/screen_action_labels/data/Wonderbread/gold_demos/0 @ 2023-12-25-15-10-58/screenshots/2.png'
extract_ui_elements(image_path)