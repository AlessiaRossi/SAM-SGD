import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
'''
df = pd.read_csv("results/metrics_summary.csv")

# Colonne da mantenere
columns = [
    'lambda','val_loss','val_accuracy','test_loss','test_accuracy','sharpness',
    'sparsity','path_norm','stability','prediction_variance','robust_accuracy',
    'flat_minima','ece','loss_type','optimizer'
]

# Filtra solo le colonne richieste (se presenti)
columns_present = [col for col in columns if col in df.columns]
df = df[columns_present]

# Salva una tabella per ogni ottimizzatore
for optimizer, group in df.groupby('optimizer'):
    group.to_csv(f"results/tabella_metriche_{optimizer.lower()}.csv", index=False)

# (Opzionale) Stampa una preview
print(df.head())
'''

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

metrics = ['test_loss','test_accuracy','path_norm','stability','robust_accuracy','flat_minima']

# Fixed color palette for each loss
LOSS_COLORS = {
    "focal": "#1f77b4",
    "huber": "#ff7f0e",
    "logitnorm": "#2ca02c",
    "saloss": "#d62728",
    "trades": "#9467bd",
}

def parse_tensor_string(val):
    """Extract float value from a string like 'tensor(0.1144, device='cuda:0')'."""
    if isinstance(val, str) and val.startswith("tensor("):
        try:
            return float(val.split("(")[1].split(",")[0])
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def normalize_metrics(df, metrics):
    """Normalize metrics between 0 and 1."""
    df_norm = df.copy()
    for m in metrics:
        vals = df_norm[m].apply(parse_tensor_string)
        minv, maxv = vals.min(), vals.max()
        if maxv > minv:
            df_norm[m] = (vals - minv) / (maxv - minv)
        else:
            df_norm[m] = 0.0
    return df_norm

def plot_bar_per_lambda(df, optimizer):
    df_norm = normalize_metrics(df, metrics)
    for l in sorted(df_norm['lambda'].unique()):
        subset = df_norm[df_norm['lambda'] == l]
        plt.figure(figsize=(10, 6))
        bar_width = 0.12
        x = np.arange(len(metrics))
        for idx, (loss, row) in enumerate(subset.groupby('loss_type').first().iterrows()):
            values = [row[m] if m in row else np.nan for m in metrics]
            color = LOSS_COLORS.get(loss, None)
            plt.bar(x + idx * bar_width, values, width=bar_width, label=loss, color=color, alpha=0.8)
        plt.xticks(x + bar_width * (len(subset['loss_type'].unique())-1)/2, metrics)
        plt.title(f"{optimizer.upper()} - Normalized Metrics for λ={l} \n λ*CE Loss+(1-λ)*Other Loss")
        plt.ylabel("Normalized value")
        plt.xlabel("Metrics")
        plt.legend(title="Loss type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/bar_{optimizer}_lambda_{l}_norm.png")
        plt.close()

def plot_metric_vs_lambda(df, optimizer):
    df_norm = normalize_metrics(df, metrics)
    for m in metrics:
        plt.figure(figsize=(10, 6))
        for loss in df_norm['loss_type'].unique():
            subset = df_norm[df_norm['loss_type'] == loss]
            color = LOSS_COLORS.get(loss, None)
            plt.plot(subset['lambda'], subset[m], marker='o', label=loss, color=color, alpha=0.8)
        plt.title(f"{optimizer.upper()} - {m} vs lambda")
        plt.xlabel("lambda")
        plt.ylabel(f" {m}")
        plt.legend(title="Loss type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/metric_vs_lambda_{optimizer}_{m}_norm.png")
        plt.close()

def plot_spider_per_lambda(optimizer):
    # Metrics for spider (excluding 'sparsity' and 'sharpness')
    spider_metrics = ['test_loss','test_accuracy','path_norm','stability','robust_accuracy','flat_minima']
    orig_path = "results/metrics_summary.csv"
    if not os.path.exists(orig_path):
        print("metrics_summary.csv not found for spider plot.")
        return
    orig = pd.read_csv(orig_path)
    orig = orig[orig["optimizer"] == optimizer]
    # Normalize spider metrics
    for m in spider_metrics:
        vals = orig[m].apply(parse_tensor_string)
        minv, maxv = vals.min(), vals.max()
        if maxv > minv:
            orig[m] = (vals - minv) / (maxv - minv)
        else:
            orig[m] = 0.0
    # One spider plot for each lambda
    for l in sorted(orig['lambda'].unique()):
        subset = orig[orig['lambda'] == l]
        if subset.empty:
            continue
        categories = spider_metrics
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        plt.figure(figsize=(8, 8))
        for _, row in subset.iterrows():
            values = [row[m] if m in row else 0 for m in categories]
            values += values[:1]
            color = LOSS_COLORS.get(row['loss_type'], None)
            label = f"{row['loss_type']} (λ={row['lambda']})"
            plt.polar(angles, values, marker='o', label=label, color=color, alpha=0.7)
        plt.xticks(angles[:-1], categories)
        plt.yticks([])  # Remove numbers from circles
        plt.title(f"{optimizer.upper()} - Spider plot (normalized) for λ={l} \n λ*CE Loss+(1-λ)*Other Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/spider_{optimizer}_lambda_{l}_norm.png")
        plt.close()

# Main loop for each optimizer
for optimizer in ["sam", "sgd"]:
    csv_path = f"results/tabella_metriche_{optimizer}.csv"
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found.")
        continue
    df = pd.read_csv(csv_path)
    df['lambda'] = df['lambda'].astype(float)
    plot_bar_per_lambda(df, optimizer)
    plot_metric_vs_lambda(df, optimizer)
    plot_spider_per_lambda(optimizer)

print("Normalized plots saved in results/")

# NOTE:
# - Each graph has different colors for each loss or combination.
# - The labels and titles specify optimizer, lambda, and loss.
# - For configuration: if using loss combinations, add a note in the title or legend.
# - The formula self.lambda_ * loss1_value + (1 - self.lambda_) * loss2_value should be specified in the graph description if necessary.
#lambda,val_loss,val_accuracy,test_loss,test_accuracy,sharpness,sparsity,path_norm,stability,prediction_variance,robust_accuracy,flat_minima,ece,loss_type

