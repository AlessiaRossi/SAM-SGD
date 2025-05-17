import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_pareto_front(
    metrics,
    x_metric='prediction_variance',
    y_metric='robust_accuracy',
    color_metric='flat_minima',
    loss_names=None,
    title="Pareto Front: Trade-off Incertezza vs Robustezza",
    output_path="pareto_front.png"
):
    """
    Mostra la Pareto Front minimizzando x_metric e massimizzando y_metric,
    colorando i punti in base a color_metric (es. flat_minima).
    Se loss_names è fornito (lista di stringhe), le include nel titolo.
    """
    # Se ho loss da inserire, le appendo nel titolo
    if loss_names:
        title = f"{title}\nLoss: {', '.join(loss_names)}"

    x = np.array([m[x_metric] for m in metrics])
    y = np.array([m[y_metric] for m in metrics])
    c = np.array([m[color_metric] for m in metrics])

    # calcolo frontiera (x min, y max)
    pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        pareto[i] = not np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
    px, py = x[pareto], y[pareto]
    idx = np.argsort(px)
    px, py = px[idx], py[idx]

    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, y, c=c, cmap='viridis', s=80, alpha=0.8)
    plt.plot(px, py, 'r-', lw=2, label="Pareto Front")
    for m in metrics:
        plt.annotate(f"{m['lambda']:.2f}",
                     (m[x_metric], m[y_metric]),
                     textcoords="offset points", xytext=(3,3), fontsize=8)
    plt.colorbar(sc, label=color_metric)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Pareto front salvata in: {output_path}")


def plot_spider(
    metrics,
    config_labels,
    metric_names=None,
    invert_metrics=None,
    loss_names=None,
    title_prefix="Spider Plot: Confronto multi-metrica",
    output_path="spider_plot.png"
):
    """
    Confronta le configurazioni su un set di metriche selezionate.
    Le metriche 'low is good' vengono invertite.
    Se loss_names è fornito, le include nel titolo.
    """
    # Metriche di default
    if metric_names is None:
        metric_names = ["val_accuracy", "test_loss", "sharpness", "sparsity"]

    # Costruisci il titolo
    title = title_prefix
    if loss_names:
        losses_str = ", ".join(loss_names)
        title = f"{title_prefix}\nLoss: {losses_str}"

    # Prepara i dati per lo Spider Plot
    def convert_to_float(value):
        if isinstance(value, torch.Tensor):  # Se è un tensor, estrai il valore
            return value.item()
        return float(value)  # Converte in float se non lo è già

    data = np.array(
        [[convert_to_float(m[name]) for name in metric_names] for m in metrics],
        dtype=float
    )

    # Normalizza i dati se necessario
    if invert_metrics:
        for i, metric in enumerate(metric_names):
            if metric in invert_metrics:
                data[:, i] = 1.0 / data[:, i]

    # Prepara gli angoli per lo Spider Plot
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Chiudi il cerchio

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in enumerate(data):
        values = row.tolist()
        values += values[:1]  # Chiudi il cerchio
        ax.plot(angles, values, label=config_labels[i])
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_title(title, size=20, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(output_path)  # Salva il grafico come file immagine
    print(f"[✓] Spider Plot salvato in: {output_path}")
    plt.close()
