import os
import random
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import argparse

### Prepare data for training and testing
# source_path: path to the source data folder
# target_path: path to the target data folder
# train_ratio: ratio of training data
# test_ratio: ratio of testing data
# return: path to the training and testing data folders

# This simple script splits the data into training and testing data.

def prepare_data(source_path, target_path, train_ratio=0.1, test_ratio=0.1):
    # if target_path does not exist, create it
    os.path.exists(target_path) or os.makedirs(target_path)
    save_path = Path(target_path) / f'split_{time.time()}_train_{train_ratio}'
    save_path_training = save_path / 'training'
    save_path_testing = save_path / 'testing'
    
    all_videos = os.listdir(source_path)
    random.seed(42)
    random.shuffle(all_videos)
    
    total_videos_number = len(all_videos)
    assert 0 < train_ratio < total_videos_number
    
    if train_ratio < 1 and test_ratio < 1:
        train_data = all_videos[:int(total_videos_number * train_ratio)]
        if test_ratio > 0:
            test_data = all_videos[int(total_videos_number * train_ratio):int(total_videos_number * (train_ratio + test_ratio))]
        else:
            test_data = all_videos[int(total_videos_number * train_ratio):]
    elif train_ratio > 1:
        train_data = all_videos[:train_ratio]
        if test_ratio > 0:
            test_data = all_videos[train_ratio:train_ratio + test_ratio]
        else:
            test_data = all_videos[train_ratio:]
    else:
        raise ValueError("Invalid train_ratio or test_ratio")
    
    save_path_training.mkdir(parents=True, exist_ok=True)
    for video in tqdm(train_data, desc="Copying training data"):
        video_path = Path(source_path) / video / 'screenshots'
        if video_path.exists():
            shutil.copytree(video_path, save_path_training / video)
    
    save_path_testing.mkdir(parents=True, exist_ok=True)
    for video in tqdm(test_data, desc="Copying testing data"):
        video_path = Path(source_path) / video / 'screenshots'
        if video_path.exists():
            shutil.copytree(video_path, save_path_testing / video)
    
    return save_path_training, save_path_testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training and testing.")
    parser.add_argument("source_path", type=str, help="Path to the source data folder.")
    parser.add_argument("target_path", type=str, help="Path to the target data folder.")
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Ratio of training data.")
    parser.add_argument("--test_ratio", type=float, default=0.9, help="Ratio of testing data.")
    
    args = parser.parse_args()
    
    prepare_data(args.source_path, args.target_path, args.train_ratio, args.test_ratio)
