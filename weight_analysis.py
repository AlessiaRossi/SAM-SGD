# weight_analysis.py (modificato: aggiunto Stability Score e varianza delle predizioni)
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model.Net import WRN56_2, WRN56_4, WRN56_8


def compute_sparsity_and_pathnorm(model):
    total_params = 0
    zero_params = 0
    pathnorm_squared = 0.0
    with torch.no_grad():
        for param in model.parameters():
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
            pathnorm_squared += torch.norm(param, p=2).item() ** 2
    sparsity = zero_params / total_params
    pathnorm = pathnorm_squared ** 0.5
    return sparsity, pathnorm


def compute_stability_score(model, inputs, N=5):
    """
    Calcola lo Stability Score (mediana della varianza sui logits) e la varianza delle predizioni.

    Args:
        model (torch.nn.Module): Il modello da analizzare.
        inputs (torch.Tensor): Gli input su cui calcolare la stabilità.
        N (int): Numero di run per calcolare la varianza.

    Returns:
        float: Mediana della varianza sui logits.
        float: Varianza delle predizioni.
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
    sparsities = []
    pathnorms = []
    stability_scores = []
    prediction_variances = []
    all_weights = []

    print("\n[Analisi pesi per lambda]")
    print(f"{'Lambda':>8} | {'Sparsity':>10} | {'PathNorm':>10} | {'Stability':>10} | {'PredVar':>10}")
    print("-" * 60)

    for lam in lambdas:
        model_path = os.path.join(models_dir, f"model_sgd_{lam:.2f}.pth")
        if not os.path.exists(model_path):
            print(f"[SKIP] Model {model_path} non trovato.")
            continue

        model = load_model(model_fn, model_path, device)
        s, p = compute_sparsity_and_pathnorm(model)
        sparsities.append((lam, s))
        pathnorms.append((lam, p))

        # Calcola Stability Score e varianza delle predizioni
        median_logits_var, mean_pred_var = compute_stability_score(model, inputs, N)
        stability_scores.append((lam, median_logits_var))
        prediction_variances.append((lam, mean_pred_var))

        # Raccogli tutti i pesi per histogramma aggregato
        weights = torch.cat([param.view(-1).detach().cpu() for param in model.parameters()])
        all_weights.append((lam, weights))

        print(f"{lam:>8.2f} | {s:.6f} | {p:.4f} | {median_logits_var:.6f} | {mean_pred_var:.6f}")

    return sparsities, pathnorms, stability_scores, prediction_variances, all_weights


def plot_metric_trend(values, title, ylabel, filename):
    lambdas, metrics = zip(*values)
    plt.figure()
    plt.plot(lambdas, metrics, marker='o')
    plt.title(title)
    plt.xlabel("Lambda")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", filename))
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

    sparsities, pathnorms, stability_scores, prediction_variances, all_weights = analyze_and_plot(
        args.models_dir, model_fn, lambdas, device, inputs, args.N
    )
    plot_metric_trend(sparsities, "Sparsity vs Lambda", "Sparsity", "sparsity_vs_lambda.png")
    plot_metric_trend(pathnorms, "PathNorm vs Lambda", "PathNorm", "pathnorm_vs_lambda.png")
    plot_metric_trend(stability_scores, "Stability Score vs Lambda", "Stability Score", "stability_vs_lambda.png")
    plot_metric_trend(prediction_variances, "Prediction Variance vs Lambda", "Prediction Variance", "prediction_variance_vs_lambda.png")
    plot_aggregated_weight_distributions(all_weights)
