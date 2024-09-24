import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "/home/moucheng/projects/screen_action_labels/results/1725541328/evals1725734866/results_evals_full.csv"
df = pd.read_csv(results_dir)

# Function to create and save violin and swarm plots
def create_violin_swarm_plot(df, metric, filename):
    df_metric = df[['demo_name', metric, 'n_lines_pred_sop', 'n_lines_gold_sop']]
    df_metric_sorted = df_metric.sort_values(by='n_lines_gold_sop')
    df_metric_sorted['n_lines_gold_sop'] = df_metric_sorted['n_lines_gold_sop'].astype(int)
    df_metric_sorted[metric] = df_metric_sorted[metric].astype(float)

    # Filter out invalid data
    df_metric_sorted = df_metric_sorted[(df_metric_sorted['n_lines_gold_sop'] > 0) & (df_metric_sorted[metric] >= 0) & (df_metric_sorted[metric] <= 1.0)]

    # Define bins
    bins = [1, 6, 11, 16, 21, float('inf')]
    labels = ['1-5', '6-10', '11-15', '16-20', '20+']

    # Bin 'n_lines_gold_sop' into defined intervals
    df_metric_sorted['n_lines_gold_sop_bin'] = pd.cut(df_metric_sorted['n_lines_gold_sop'], bins=bins, labels=labels, right=False)

    # Verify the filtered and binned data
    print(f"Filtered Data Summary for {metric}:")
    print(df_metric_sorted.describe())
    print(f"Binned Data Summary for {metric}:")
    print(df_metric_sorted['n_lines_gold_sop_bin'].value_counts())

    # Plotting the results as a violin plot combined with a swarm plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'{metric.capitalize()} Distribution by Binned No. lines of Gold SOP')
    ax.set_xlabel('Number of lines of Gold SOP (Binned)')
    ax.set_ylabel(metric.capitalize())

    # Create violin plot
    sns.violinplot(x='n_lines_gold_sop_bin', y=metric, data=df_metric_sorted, ax=ax, inner='quartile', palette='Set2')
    # Create swarm plot with size reflecting the number of cases
    sns.swarmplot(x='n_lines_gold_sop_bin', y=metric, data=df_metric_sorted, ax=ax, size=5, color='black', alpha=0.6)
    # Add y axis = 0 and y axis = 1 lines:
    ax.axhline(y=0, color='r', linestyle='--')
    ax.axhline(y=1, color='r', linestyle='--')
    # Show and save the plot
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

# Create plots for precision, recall, and ordering
create_violin_swarm_plot(df, 'precision', '../figures/precision_distribution_violin_swarm_plot.png')
create_violin_swarm_plot(df, 'recall', '../figures/recall_distribution_violin_swarm_plot.png')
create_violin_swarm_plot(df, 'ordering', '../figures/ordering_distribution_violin_swarm_plot.png')