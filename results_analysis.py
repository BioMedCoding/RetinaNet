import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to generate plots without needing a display server
import matplotlib.pyplot as plt
import os

# =====================================================================================================
'''

GENERAL INFO
This code is created as a semi-autonomous system to analyse the results created in different trials and rank them.
The file indicated in file_path is the one generated by the script "evaluation.py" and so requires a rigid, fixed structure and can not work on any other file.

HOW TO USE
1. Update the file_path variable to point to the file created by the "evaluation.py" script.
2. Specify the number of best trials to analyze and display in the graphs.
3. Execute the script.
4. Upon completion, three files will be generated:
    - dice_rank: ranks each trial by Dice value, in descending order (higher values are better).
    -  error_rank: ranks each trial by error value, in ascending order (lower values are better, combining J2J and J2E errors compared to the ground truth mask).
    - final_comparison: Merges the two previous rankings to highlight trials with the best overall performance.
5. Two graphs will be generated for the top n trials:
    - The first graph displays the Dice values.
    - The second graph shows both J2E and J2J errors.
   
'''
# =====================================================================================================

# =====================================================================================================
# Variables definitions
# =====================================================================================================

working_folder = os.path.dirname(os.path.abspath(__file__))
file_path = working_folder+'/Output_postprocess_finale/results.txt'

n = 5 # Number of best trials to analyze and display in the graphs.

matplotlib.use('Agg')  # Set the backend to 'Agg' to generate plots without needing a display server


# Function to plot and save the n best trials based on dice values with grid and numeric values
def plot_best_dice_values(df, n):
    df_top_n_dice = df.nlargest(n, 'mean_dice_total')
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_top_n_dice['experiment_id'], df_top_n_dice['mean_dice_total'], color='skyblue')
    plt.xlabel('Experiment ID')
    plt.ylabel('Dice Value')
    plt.title(f'Top {n} Trials based on Dice Values')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')  # Add grid on y-axis with dashed lines
    
    # Annotate bars with numeric values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(results_folder + '/top_n_dice_values.png')
    plt.close()

# Function to plot and save the n best trials based on total mean J2E and J2J errors with grid and numeric values
def plot_best_errors(df, n):
    df['total_mean_error'] = df['mean_diff_N_J2E'] + df['mean_diff_N_J2J']
    df_top_n_errors = df.nsmallest(n, 'total_mean_error')
    plt.figure(figsize=(10, 6))
    width = 0.35  # the width of the bars
    r1 = range(len(df_top_n_errors))
    r2 = [x + width for x in r1]

    bars_j2e = plt.bar(r1, df_top_n_errors['mean_diff_N_J2E'], width, label='J2E Error', color='salmon')
    bars_j2j = plt.bar(r2, df_top_n_errors['mean_diff_N_J2J'], width, label='J2J Error', color='lightgreen')
    
    plt.xlabel('Experiment ID')
    plt.ylabel('Error Value')
    plt.title(f'Top {n} Trials based on J2E and J2J Errors')
    plt.xticks([r + width/2 for r in range(len(df_top_n_errors))], df_top_n_errors['experiment_id'], rotation=45)
    plt.grid(axis='y', linestyle='--')  # Add grid on y-axis with dashed lines
    plt.legend()

    # Annotate bars with numeric values for both J2E and J2J errors
    for bars in [bars_j2e, bars_j2j]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(results_folder + '/top_n_j2e_j2j_errors.png')
    plt.close()

# Create a dictionary to save the different evaluation metrics
data_dict = {
    "experiment_id": [],
    "mean_dice_total": [],
    "mean_diff_N_J2E": [],
    "mean_diff_N_J2J": []
}

# Process the file to extract all required metrics
with open(file_path, 'r') as file:
    experiment_blocks = file.read().split('\n\n')

for block in tqdm(experiment_blocks):
    lines = block.split('\n')
    experiment_id = lines[0].replace(':', '').strip()
    metrics = {line.split(':')[0]: float(line.split(':')[1]) for line in lines[1:] if ':' in line}
    if 'mean_dice_total' in metrics and 'mean_diff_N_J2E' in metrics and 'mean_diff_N_J2J' in metrics:
        data_dict['experiment_id'].append(experiment_id)
        data_dict['mean_dice_total'].append(metrics['mean_dice_total'])
        data_dict['mean_diff_N_J2E'].append(metrics['mean_diff_N_J2E'])
        data_dict['mean_diff_N_J2J'].append(metrics['mean_diff_N_J2J'])

# Creating a DataFrame from the extracted data
df_parsed_complete = pd.DataFrame(data_dict)


# Use df_parsed_complete as DataFrame for analysis
df = df_parsed_complete.copy()

# Calculate the sum of mean_diff_N_J2E and mean_diff_N_J2J
df['sum_diff'] = df['mean_diff_N_J2E'] + df['mean_diff_N_J2J']

# Ranking based on mean_dice_total
df_dice_rank = df.sort_values(by="mean_dice_total", ascending=False)
dice_rank_text = "\n".join([f"{row['experiment_id']}: {row['mean_dice_total']}" for _, row in df_dice_rank.iterrows()])

# Ranking based on the sum of mean_diff_N_J2E and mean_diff_N_J2J, with details
df_diff_rank = df.sort_values(by="sum_diff")
diff_rank_text_with_values = "\n".join([
    f"{row['experiment_id']}: {row['sum_diff']} (J2E: {row['mean_diff_N_J2E']}, J2J: {row['mean_diff_N_J2J']})" 
    for _, row in df_diff_rank.iterrows()
])

# Calculate the complemented dice
df['metric_1_minus_dice'] = 1 - df['mean_dice_total']

# Creation of the new general metric, to evaluate the overall performance
df['final_metric_sum'] = df['metric_1_minus_dice'] + df['sum_diff']

df_final_comparison = df.sort_values(by="final_metric_sum")
final_comparison_text = "\n".join([
    f"{row['experiment_id']}: {row['final_metric_sum']}" 
    for _, row in df_final_comparison.iterrows()
])

# Paths for the output files
results_folder = working_folder+'/auto_results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

dice_rank_file_path = results_folder + '/dice_rank.txt'
diff_rank_file_path = results_folder + '/error_rank.txt'
final_comparison_file_path = results_folder + '/final_comparison.txt'

# Writing the files
with open(dice_rank_file_path, 'w') as file:
    file.write(dice_rank_text)

with open(diff_rank_file_path, 'w') as file:
    file.write(diff_rank_text_with_values)

with open(final_comparison_file_path, 'w') as file:
    file.write(final_comparison_text)

print(f"Files created:\n{dice_rank_file_path}\n{diff_rank_file_path}\n{final_comparison_file_path}")

# Sezione per creare i grafici 
plot_best_dice_values(df, n)
plot_best_errors(df, n)