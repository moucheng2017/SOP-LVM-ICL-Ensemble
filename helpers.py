import os
import cv2
import base64
from collections import Counter
import yaml

def check_videos_paths(screenshots_paths, labels_paths):
    if len(screenshots_paths) != len(labels_paths):
        raise ValueError(f"Number of screenshots paths ({len(screenshots_paths)}) and labels paths ({len(labels_paths)}) do not match.")
    # make sure for each paired screenshots path and labels path, the video name is the same
    for i in range(len(screenshots_paths)):
        if screenshots_paths[i].split('/')[-2] != labels_paths[i].split('/')[-2]:
            raise ValueError(f"Video names do not match: {screenshots_paths[i].split('/')[-2]} and {labels_paths[i].split('/')[-2]}")
    print("Videos paths are consistent in paired screenshots and labels.")

def read_paths_from_txt(txt_path):
    # print('txt_path: ', txt_path)   
    with open(txt_path, 'r') as f:
        paths = f.readlines()
    paths_list = [path.strip() for path in paths]
    # sort the paths to make sure the order is consistent
    paths_list.sort()
    # print(paths_list)
    return paths_list

def save_config(config, save_path):
    with open(save_path / 'config.yaml', 'w') as file:
        yaml.dump(config, file)
        
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def sort_key(filename):
    # check if the filename is in the format of "0.png":
    if filename.split('.')[0].isdigit():
        return int(filename.split('.')[0])
    else:
        return int(filename.split('_')[1].split('.')[0])

def get_screenshots(folder_path):
    screenshots_folder_path = os.path.join(folder_path, 'screenshots')
    if not os.path.exists(screenshots_folder_path):
        screenshots_folder_path = f'"{screenshots_folder_path}"' 
    else:
        pass

    screenshots = sorted([f for f in os.listdir(screenshots_folder_path) if f.endswith('.png')], key=sort_key)
    print(f"Found {len(screenshots)} screenshots in {screenshots_folder_path}")
    if len(screenshots) == 0:
        print(f"No screenshots found in {screenshots_folder_path}")
    elif len(screenshots) > 15:
        # downsample to 15 frames with equal intervals
        screenshots = [screenshots[i] for i in range(0, len(screenshots), len(screenshots)//15)]
    return screenshots, screenshots_folder_path
    
def read_frames(folder_path, resize=None):
    base64_frames = []
    screenshots, screenshots_folder_path = get_screenshots(folder_path)
    if len(screenshots) > 15:
        screenshots = [screenshots[i] for i in range(0, len(screenshots), len(screenshots)//15)]
    # for gpt:
    for f in screenshots:
        img = cv2.imread(os.path.join(screenshots_folder_path, f))
        if resize:
            img = cv2.resize(img, tuple(resize))
        _, buffer = cv2.imencode('.png', img)
        base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    return base64_frames, len(screenshots)

def read_labels(folder_path):
    label_path = [f for f in os.listdir(folder_path) if f.endswith('.txt')][0]
    label_path = os.path.join(folder_path, label_path)
    try:
        labels = open(label_path, 'r').read()
        # print(f"Read labels from {label_path}:")
        # print(labels)
        return labels
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return None

def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]

def prediction_template(num_frames):
    return '\n'.join([f'{i+1}. ' for i in range(num_frames)])
