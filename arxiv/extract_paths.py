from pathlib import Path
import argparse
from tqdm import tqdm
import os

source_path = '/home/moucheng/data/Notepad/many_shots_1719414312.8246899_train_10_debug_only_detailed_labels'
source_path_train = Path(source_path) / 'training'
source_path_test = Path(source_path) / 'testing'
all_train_videos = os.listdir(source_path_train)
all_test_videos = os.listdir(source_path_test)
train_videos_paths = [os.path.join(source_path_train, video) for video in all_train_videos]
test_videos_paths = [os.path.join(source_path_test, video) for video in all_test_videos]
train_videos_screenshots = [os.path.join(source_path_train, video, 'screenshots') for video in all_train_videos]
test_videos_screenshots = [os.path.join(source_path_test, video, 'screenshots') for video in all_test_videos]
train_videos_labels = [os.path.join(source_path_train, video, 'label.txt') for video in all_train_videos]
test_videos_labels = [os.path.join(source_path_test, video, 'label.txt') for video in all_test_videos]

target_path_train = '/home/moucheng/data/split/Notepad/many_shots_1719414312.8246899_train_10_debug_only_detailed_labels/training'
target_path_test = '/home/moucheng/data/split/Notepad/many_shots_1719414312.8246899_train_10_debug_only_detailed_labels/testing'
target_train_screenshots_txt = Path(target_path_train) / 'screenshots.txt'
target_test_screenshots_txt = Path(target_path_test) / 'screenshots.txt'
target_train_labels_txt = Path(target_path_train) / 'labels.txt'
target_test_labels_txt = Path(target_path_test) / 'labels.txt'

with open(target_train_screenshots_txt, 'w') as f:
    for video in train_videos_screenshots:
        f.write(video + '\n')

with open(target_test_screenshots_txt, 'w') as f:
    for video in test_videos_screenshots:
        f.write(video + '\n')

with open(target_train_labels_txt, 'w') as f:
    for video in train_videos_labels:
        f.write(video + '\n')

# with open(target_test_labels_txt, 'w') as f:
#     for video in test_videos_labels:
#         with open(video, 'r') as f2:
#             f.write(f2.read())

print('Done')


