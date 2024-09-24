# A simple script to read two folders and list out the videos in each folder and compare them
# list out the videos that are in both folders and the videos that are in one folder but not in the other
import os
import sys

folder1 = "/home/moucheng/results/1722441353/Wonderbread/gold_demos"
folder2 = "/home/moucheng/results/1722520086/Wonderbread/gold_demos"

all_videos1 = os.listdir(folder1)
all_videos2 = os.listdir(folder2)

print('The number of videos in folder1: ', len(all_videos1))
print('The number of videos in folder2: ', len(all_videos2))

# get the videos that are in both folders
common_videos = set(all_videos1) & set(all_videos2)
print('The number of videos that are in both folders: ', len(common_videos))

# get the videos that are in folder1 but not in folder2
videos_only_in_folder1 = set(all_videos1) - set(all_videos2)
print('The number of videos that are only in folder1: ', len(videos_only_in_folder1))

# get the videos that are in folder2 but not in folder1
videos_only_in_folder2 = set(all_videos2) - set(all_videos1)
print('The number of videos that are only in folder2: ', len(videos_only_in_folder2))

