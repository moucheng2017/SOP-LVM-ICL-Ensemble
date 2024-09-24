# This script split the training data into batches and save each batch as a separate txt file
# Input 1: training_data_txt_full.txt
# Input 2: exclusion data_txt_full.txt
# Output: training_data_batches

import os
import random
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import argparse

train_txt_full = "../data_splits/gold_demos/training/screenshots.txt"
exclusion_txt_full = None
batch_size = 8
random_seed = 42

# Make the save folder for the training data batches:
save_folder = 'training_data_bs' + str(batch_size) + '_sd' + str(random_seed)
save_path = Path(train_txt_full).parent.parent / save_folder
os.makedirs(save_path, exist_ok=True)
print('train save path :', save_path)

# Load the training data txt file:
with open(train_txt_full, 'r') as f:
    train_data = f.readlines()
train_data = [x.strip() for x in train_data]

if exclusion_txt_full:
    # Load the exclusion data txt file:
    with open(exclusion_txt_full, 'r') as f:
        exclusion_data = f.readlines()
    exclusion_data = [x.strip() for x in exclusion_data]

    assert len(exclusion_data) == batch_size

    # Remove the exclusion data from the training data:
    train_data = [x for x in train_data if x not in exclusion_data]

# Randomly shuffle training data:
random.seed(random_seed)
random.shuffle(train_data)

# Split the training data into chunks of batch_size:
# print the size of the training data
print(f"Total number of videos: {len(train_data)}")
train_data_batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
print(f"Total number of batches: {len(train_data_batches)}")

for i, batch in tqdm(enumerate(train_data_batches)):
    batch_path = save_path / f'batch_{i}.txt'
    # print the number of videos in each batch:
    print(f"Number of videos in batch {i}: {len(batch)}")
    with open(batch_path, 'w') as f:
        for item in batch:
            f.write("%s\n" % item)
    print(f"Batch {i} saved to {batch_path}")

print("Done!")

