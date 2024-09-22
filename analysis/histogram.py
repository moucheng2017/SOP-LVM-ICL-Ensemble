import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "/home/moucheng/projects/screen_action_labels/results/1725541328/evals1725734866/results_evals_full.csv"
df = pd.read_csv(results_dir)

# get the subset 1 with the columns we are interested in: demo_name, precision, n_lines_pred_sop, n_lines_gold_sop
df_precision = df[['demo_name', 'precision', 'n_lines_pred_sop', 'n_lines_gold_sop']]
# get the subset 2 with the columns we are interested in: demo_name, recall, n_lines_pred_sop, n_lines_gold_sop
df_recall = df[['demo_name', 'recall', 'n_lines_pred_sop', 'n_lines_gold_sop']]
# get the subset 3 with the columns we are interested in: demo_name, ordering, n_lines_pred_sop, n_lines_gold_sop
df_ordering = df[['demo_name', 'ordering', 'n_lines_pred_sop', 'n_lines_gold_sop']]

# Function to add horizontal and vertical lines
def add_lines(ax, data, column):
    y_max = ax.get_ylim()[1]
    y_mid = y_max / 2
    hist_data = data[column].value_counts().sort_index().cumsum()
    x_intersect = hist_data[hist_data >= y_mid].index[0]
    
    # Add horizontal line up to the intersection point
    ax.plot([0, x_intersect], [y_mid, y_mid], 'r--')
    # Add vertical line from the intersection point downwards
    ax.plot([x_intersect, x_intersect], [0, y_mid], 'r--')

# plot the accumulated histograms in a row in a single frame. each with a title:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Accumulated Histograms of Precision, Recall, Ordering')
axs[0].set_title('Precision')
axs[1].set_title('Recall')
axs[2].set_title('Ordering')
df_precision['precision'].hist(ax=axs[0], bins=100, cumulative=True)
df_recall['recall'].hist(ax=axs[1], bins=100, cumulative=True)
df_ordering['ordering'].hist(ax=axs[2], bins=100, cumulative=True)

# Add lines
add_lines(axs[0], df_precision, 'precision')
add_lines(axs[1], df_recall, 'recall')
add_lines(axs[2], df_ordering, 'ordering')

# Ensure the axes intersect at the origin (0,0)
for ax in axs:
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.show()
# save the plot
fig.savefig('accumulated_histograms.png', bbox_inches='tight')

