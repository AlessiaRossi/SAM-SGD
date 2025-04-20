# weight_analysis.py (modificato: aggiunto Stability Score e varianza delle predizioni)
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model.Net import WRN56_2, WRN56_4, WRN56_8


def compute_sparsity_and_pathnorm(model):
    """
    Calcola la sparsità e il PathNorm del modello.

    Returns:
        tuple: Sparsità e PathNorm.
    """
    total_params = 0
    zero_params = 0
    path_norm = 0.0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
        path_norm += param.norm(2).item()

    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return sparsity, path_norm


def compute_stability_score(model, inputs, N=5):
    """
    Calcola la stabilità del modello perturbando i pesi.

    Args:
        model (torch.nn.Module): Modello.
        inputs (torch.Tensor): Input per il calcolo.
        N (int): Numero di perturbazioni.

    Returns:
        tuple: Stabilità e varianza delle predizioni.
    """
    logits_list = []
    predictions_list = []

    model.eval()
    with torch.no_grad():
        for _ in range(N):
            outputs = model(inputs)
            logits_list.append(outputs.cpu().numpy())
            predictions_list.append(outputs.argmax(dim=1).cpu().numpy())

    logits_array = np.stack(logits_list, axis=0)  # Shape: (N, batch_size, num_classes)
    predictions_array = np.stack(predictions_list, axis=0)  # Shape: (N, batch_size)

    # Calcola la varianza sui logits lungo l'asse delle run (N)
    logits_variance = np.var(logits_array, axis=0)  # Shape: (batch_size, num_classes)
    median_logits_variance = np.median(logits_variance)  # Mediana della varianza sui logits

    # Calcola la varianza delle predizioni lungo l'asse delle run (N)
    predictions_variance = np.var(predictions_array, axis=0)  # Shape: (batch_size,)
    mean_predictions_variance = np.mean(predictions_variance)  # Media della varianza delle predizioni

    return median_logits_variance, mean_predictions_variance


def load_model(model_fn, model_path, device):
    model = model_fn()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def analyze_and_plot(models_dir, model_fn, lambdas, device, inputs, N=5):
    """
    Analizza i pesi per diversi valori di lambda e stampa i risultati.

    Args:
        models_dir (str): Directory dei modelli salvati.
        model_fn (callable): Funzione per caricare il modello.
        lambdas (list): Lista di valori di lambda.
        device (torch.device): Dispositivo (CPU o GPU).
        inputs (torch.Tensor): Input per il calcolo della stabilità.
        N (int): Numero di perturbazioni per la stabilità.

    Returns:
        tuple: Sparsità, PathNorm, Stabilità, PredVar.
    """
    sparsities = []
    pathnorms = []
    stability_scores = []
    prediction_variances = []

    print("\n[Analisi pesi per lambda]")
    print(f"{'Lambda':>8} | {'Sparsity':>10} | {'PathNorm':>10} | {'Stability':>10} | {'PredVar':>10}")
    print("-" * 58)

    for lam in lambdas:
        model_path = os.path.join(models_dir, f"model_sgd_{lam:.2f}.pth")
        if not os.path.exists(model_path):
            print(f"Modello non trovato per lambda={lam:.2f}")
            continue

        model = load_model(model_fn, model_path, device)
        sparsity, path_norm = compute_sparsity_and_pathnorm(model)
        stability, pred_var = compute_stability_score(model, inputs, N)

        # Aggiungi i risultati come tuple (lambda, valore)
        sparsities.append((lam, sparsity))
        pathnorms.append((lam, path_norm))
        stability_scores.append((lam, stability))
        prediction_variances.append((lam, pred_var))

        print(f"{lam:>8.2f} | {sparsity:>10.6f} | {path_norm:>10.4f} | {stability:>10.6f} | {pred_var:>10.6f}")

    return sparsities, pathnorms, stability_scores, prediction_variances


def plot_metric_trend(values, title, ylabel, output_file):
    """
    Plotta l'andamento di una metrica rispetto a lambda.

    Args:
        values (list of tuple): Lista di tuple (lambda, metrica).
        title (str): Titolo del grafico.
        ylabel (str): Etichetta dell'asse y.
        output_file (str): Nome del file di output per salvare il grafico.
    """
    # Verifica che values sia una lista di tuple
    if not isinstance(values, list) or not all(isinstance(v, tuple) and len(v) == 2 for v in values):
        raise ValueError("Il parametro 'values' deve essere una lista di tuple (lambda, metrica).")

    lambdas, metrics = zip(*values)  # Estrae i valori di lambda e le metriche
    plt.figure()
    plt.plot(lambdas, metrics, marker='o')
    plt.title(title)
    plt.xlabel("Lambda")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_aggregated_weight_distributions(all_weights):
    plt.figure()
    for lam, weights in all_weights:
        plt.hist(weights.numpy(), bins=100, alpha=0.4, label=f"λ={lam:.2f}", density=True)
    plt.title("Distribuzione aggregata dei pesi per lambda")
    plt.xlabel("Valore dei pesi")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "aggregated_weight_distributions.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--lambda_range", default="0,1,0.2")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--N", type=int, default=5, help="Numero di run per calcolare lo Stability Score")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fn = {2: WRN56_2, 4: WRN56_4, 8: WRN56_8}.get(args.depth)
    if model_fn is None:
        raise ValueError("Profondità WRN non valida")

    start, end, step = map(float, args.lambda_range.split(","))
    lambdas = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]

    # Genera input casuali per calcolare lo Stability Score
    inputs = torch.randn(args.batch_size, 3, 32, 32).to(device)

    sparsities, pathnorms, stability_scores, prediction_variances = analyze_and_plot(
        models_dir="results/",
        model_fn=model_fn,
        lambdas=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        device=device,
        inputs=inputs,
        N=5
    )

    # Plotta le metriche
    plot_metric_trend(sparsities, "Sparsity vs Lambda", "Sparsity", "sparsity_vs_lambda.png")
    plot_metric_trend(pathnorms, "PathNorm vs Lambda", "PathNorm", "pathnorm_vs_lambda.png")
    plot_metric_trend(stability_scores, "Stability vs Lambda", "Stability", "stability_vs_lambda.png")
    plot_metric_trend(prediction_variances, "Prediction Variance vs Lambda", "Prediction Variance", "predvar_vs_lambda.png")
