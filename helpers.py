import os
import cv2
import base64
from collections import Counter
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def sort_key(filename):
    return int(filename.split('_')[1].split('.')[0])

def read_frames(folder_path, resize=None):
    screenshots_folder_path = os.path.join(folder_path, 'screenshots')
    base64_frames = []
    screenshots = sorted([f for f in os.listdir(screenshots_folder_path) if f.endswith('.png')], key=sort_key)
    
    for f in screenshots:
        img = cv2.imread(os.path.join(screenshots_folder_path, f))
        if resize:
            img = cv2.resize(img, tuple(resize))
        _, buffer = cv2.imencode('.png', img)
        base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    return base64_frames, len(screenshots)

def read_labels(folder_path):
    label_path = os.path.join(folder_path, 'label.txt')
    try:
        labels = open(label_path, 'r').read()
        return labels
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return None

def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]

def prediction_template(num_frames):
    return '\n'.join([f'State {i}:' for i in range(num_frames)])
