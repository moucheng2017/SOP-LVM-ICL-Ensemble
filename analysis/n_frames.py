import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Study the number of frames in the data and the relationship between that and the number of lines in the SOPs for each video:
data_folder = "/home/moucheng/projects/screen_action_labels/data/Wonderbread/gold_demos"
split = "/home/moucheng/projects/screen_action_labels/data/split/Wonderbread/1723811870.4719887/testing/screenshots.txt" # a file with all the paths to the screenshots in the testing set

# Get the screenshots paths:
with open(split, 'r') as f:
    screenshots_paths = f.readlines()

# Get the video names:
video_names = [v.strip().split('/')[-2] for v in screenshots_paths]

# Get the video paths:
video_paths = [f"{data_folder}/{video}" for video in video_names]

# Get the SOP paths:
sop_paths = [f"{data_folder}/{video}/SOP*.txt" for video in video_names]

# Prepare a dataframe to store the number of frames and the number of lines in the SOPs for each video
df = pd.DataFrame(columns=['video', 'n_frames', 'n_lines'])
for i, (video_path, sop_path) in enumerate(zip(video_paths, sop_paths)):
    # Get the number of frames:
    n_frames = len([f for f in os.listdir(os.path.join(video_path, 'screenshots')) if f.endswith('.png')]) 
    # Get the number of lines in the SOPs:
    n_lines = 0
    for sop in glob.glob(sop_path):
        with open(sop, 'r') as f:
            n_lines += len(f.readlines())
    df.loc[i] = [video_names[i], n_frames, n_lines]


plt.figure(figsize=(10, 6))
# plot the histogram of the number of frames and the number of lines in the SOPs
sns.histplot(df['n_frames'], label='Number of frames', color='red')
sns.histplot(df['n_lines'], label='Number of lines in the SOPs', color='blue')

plt.xlabel('Number of frames/lines', fontdict={'fontsize': 20})
plt.ylabel('Density', fontdict={'fontsize': 20})
# plt.title('Kernel Density Estimate of the number of frames and the number of lines in the SOPs')
plt.legend(fontsize=20)
plt.show()
# save the plot
plt.savefig('../figures/n_frames_vs_n_lines.png', bbox_inches='tight')


# Add another column to the dataframe to store the ratio of the number of lines in the SOPs to the number of frames
df['ratio'] = df['n_lines'] / df['n_frames']

# Plot the distribution of the ratio
plt.figure(figsize=(10, 6))
sns.histplot(df['ratio'], color='red')
plt.xlabel('n_lines_Gold_SOP / n_frames', fontdict={'fontsize': 20})
plt.ylabel('Frequency', fontdict={'fontsize': 20})
plt.show()
# save the plot
plt.savefig('../figures/ratio.png', bbox_inches='tight')
