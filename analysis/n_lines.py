import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
results_dir = "/home/moucheng/projects/screen_action_labels/results/1725541328/evals1725734866/results_evals_full.csv"
df = pd.read_csv(results_dir)

# Plot the distributions using seaborn
plt.figure(figsize=(12, 8))
kde_pred = sns.kdeplot(df['n_lines_pred_sop'], label='n_lines_pred_sop', log_scale=True, color='blue')
kde_gold = sns.kdeplot(df['n_lines_gold_sop'], label='n_lines_gold_sop', log_scale=True, color='red')

# plt.title('Log Scale Distribution of n_lines_pred_sop and n_lines_gold_sop')
plt.xlabel('Lines of SOPs', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.legend(fontsize=20)

# Calculate the maximum density values at the mean of each distribution
mean_pred = df['n_lines_pred_sop'].mean()
mean_gold = df['n_lines_gold_sop'].mean()

max_density_pred = kde_pred.get_lines()[0].get_data()[1].max()
max_density_gold = kde_gold.get_lines()[1].get_data()[1].max()

# Add vertical lines capped at the intersection points
plt.axvline(mean_pred, color='blue', linestyle='dashed', linewidth=2, ymax=max_density_pred)
plt.axvline(mean_gold, color='red', linestyle='dashed', linewidth=2, ymax=max_density_gold)

# Add x axis values where the vertical lines are at y = 0:
plt.text(mean_pred, 0, f'{mean_pred:.2f}', fontsize=12, ha='center', va='top', color='blue')
plt.text(mean_gold, 0, f'{mean_gold:.2f}', fontsize=12, ha='center', va='top', color='red')

# Save the figure
plt.savefig('../figures/log_scale_sop_lines.png', bbox_inches='tight') 
plt.show()
