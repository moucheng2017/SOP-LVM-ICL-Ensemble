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

def prepare_data(source_path, 
                 train_ratio=0.1, 
                 test_ratio=0.1,
                 random_seed=42):
    
    # Go to the parent folder of the source_path as save_path
    save_path = Path(source_path).parent.parent
    print(f"save_path_parent: {save_path}")
    dataset_name = source_path.split('/')[-2]   
    save_path = save_path / 'split' / dataset_name / str(time.time())
    print(f"save_path_full: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    save_path_training = save_path / 'training'
    save_path_testing = save_path / 'testing'

    all_videos = os.listdir(source_path)
    # remove hidden files
    all_videos = [video for video in all_videos if not video.startswith('.')]
    # add the full path to the video
    all_videos = [os.path.join(source_path, video) for video in all_videos]
    print(f"Total number of videos: {len(all_videos)}")

    random.seed(random_seed)
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
    save_path_training_screenshots_txt = save_path_training / 'screenshots.txt'
    save_path_training_labels_txt = save_path_training / 'labels.txt'

    with open(save_path_training_screenshots_txt, 'w') as f:
        for video in tqdm(train_data):
            f.write(video + '/screenshots \n')
    
    with open(save_path_training_labels_txt, 'w') as f:
        for video in tqdm(train_data):
            f.write(video + '/*.txt \n')

    save_path_testing.mkdir(parents=True, exist_ok=True)
    save_path_testing_screenshots_txt = save_path_testing / 'screenshots.txt'
    save_path_testing_labels_txt = save_path_testing / 'labels.txt'

    with open(save_path_testing_screenshots_txt, 'w') as f:
        for video in test_data:
            f.write(video + '/screenshots \n')
    
    with open(save_path_testing_labels_txt, 'w') as f:
        for video in test_data:
            f.write(video + '/*.txt \n')
    
    # get the split information:
    print(f"Training data: {len(train_data)}")
    print(f"Testing data: {len(test_data)}")

    # save the split information to the split folder
    with open(save_path / 'split_info.txt', 'w') as f:
        f.write(f"Training data: {len(train_data)}\n")
        f.write(f"Testing data: {len(test_data)}\n")
        f.write(f"Training ratio: {train_ratio}\n")
        f.write(f"Testing ratio: {test_ratio}\n")
        f.write(f"Total number of videos: {total_videos_number}\n")
        f.write(f"Random seed: {random_seed}\n")
    
    # return save_path_training, save_path_testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training and testing.")
    parser.add_argument("--source_path", type=str, help="Path to the source data folder.", default="/home/moucheng/data/Wonderbread/gold_demos")
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Ratio of training data.")
    parser.add_argument("--test_ratio", type=float, default=0.7, help="Ratio of testing data.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    
    prepare_data(args.source_path, 
                 args.train_ratio, 
                 args.test_ratio,
                 args.random_seed)
