import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to generate plots without needing a display server
import matplotlib.pyplot as plt
import os

# =====================================================================================================
#INFO CODICE
'''
Questo codice è il codice utilizzato per analizzare in maniera semiautomatica il file contenente i risultati relativi a più trial.
Il file indicato in file_path è il file che viene generato dal codice "valutazione_rev5.py", è caratterizzato da una struttura
specifica e questo codice è infatti basato su questa specifica struttura, pertanto non può funzionare su altri file.
L'utente deve solo inserire il percorso di questo file, ottenendo 3 file al termine dell'esecuzione di questo codice:
- dice_rank: ordina i vari trial per il valore di Dice decrescent, ottenendo così il nome dei tentativi migliori
- error_rank: ordina i vari trial per valore di errore crescente (minore è l'errore, maggiore la prestazione della rete). Questo errore è calcolato come somma dei valori medi degli arrori J2E e J2J
- final_comparison: ordine la rete in ordine di prestazione complessiva decrescente (le reti migliori sono indicate per prima). Questo è fatto calcolando un indice complessivo come: (1-dice)+errore, per cui una rete perfetta ha metrica pari a 0 e cresce con il crescere degli errori o con un valore di dice peggiore

Infine vengono anche creati 2 grafici relativi alle n reti migliori (n variabile che indica il numero di reti volute) per rappresentare in un grafico il valore di dice, nell'altro il valore dei due errori (J2J e J2E). Questo permette una rapida visualizzazione delle differente
Va notato che la rete con Dice migliore non è detto che abbia anche errori minori, è per questo che si sono creati due graduatorie differenti (per la valutazione manuale) e si è creata un'apposita metrica che andasse semplicemente a valutare l'andamento complessivo
'''
# =====================================================================================================

# =====================================================================================================
# Definzione percorsi e numero di reti migliori di cui creare i grafici
# =====================================================================================================

working_folder = os.path.dirname(os.path.abspath(__file__))
#file_path = working_folder+'/Output_test/results.txt'
file_path = working_folder+'/Output_postprocess_finale/results.txt'
n = 5 # Variabile che indica quanti tentativi migliori sono da considerare per creare i grafici




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

data_dict = {
    "experiment_id": [],
    "mean_dice_total": [],
    "mean_diff_N_J2E": [],
    "mean_diff_N_J2J": []
}

# Re-processing the file to extract all required metrics
with open(file_path, 'r') as file:
    experiment_blocks = file.read().split('\n\n')

for block in tqdm(experiment_blocks):
    lines = block.split('\n')
    experiment_id = lines[0].replace(':', '').strip()
    #metrics = {line.split(': :')[0]: float(line.split(': :')[1]) for line in lines[1:] if ': :' in line}
    metrics = {line.split(':')[0]: float(line.split(':')[1]) for line in lines[1:] if ':' in line}
    if 'mean_dice_total' in metrics and 'mean_diff_N_J2E' in metrics and 'mean_diff_N_J2J' in metrics:
        data_dict['experiment_id'].append(experiment_id)
        data_dict['mean_dice_total'].append(metrics['mean_dice_total'])
        data_dict['mean_diff_N_J2E'].append(metrics['mean_diff_N_J2E'])
        data_dict['mean_diff_N_J2J'].append(metrics['mean_diff_N_J2J'])

# Creating a DataFrame from the extracted data
df_parsed_complete = pd.DataFrame(data_dict)

# Section 2 modified: analysis of the results, after import

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

# Calculate the new metric (1 - mean_dice_total) and add to DataFrame
df['metric_1_minus_dice'] = 1 - df['mean_dice_total']

# Sum of the new metric with the sum of mean_diff_N_J2E and mean_diff_N_J2J
df['final_metric_sum'] = df['metric_1_minus_dice'] + df['sum_diff']

# Final ranking based on the sum of the final metrics
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