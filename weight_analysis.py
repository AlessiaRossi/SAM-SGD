import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from model.Net import WRN56_2, WRN56_4, WRN56_8


def compute_sparsity_and_pathnorm(model):
    total_params = 0
    zero_params = 0
    path_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            path_norm += torch.norm(param, p=2).item() ** 2

    print(f"Total parameters: {total_params}, Zero parameters: {zero_params}") # per capire meglio sparcity
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    path_norm = np.sqrt(path_norm)
    print(f"Sparcity: {sparsity}, Path norm: {path_norm}")
    return sparsity, path_norm


def compute_stability_score(model,dataloader, N=5):
   
    logits_list = []
    model.eval()
    with torch.no_grad():
        for _ in range(N):
            batch_logits = []
            for inputs, _ in dataloader:
                inputs = inputs.to(next(model.parameters()).device)
                outputs = model(inputs)
                batch_logits.append(outputs.detach().cpu().numpy())
            logits_list.append(np.concatenate(batch_logits, axis=0))

    logits_array = np.stack(logits_list, axis=0)  # (N, samples, num_classes)
    stability = np.std(logits_array, axis=0).mean()
    pred_var = np.var(logits_array, axis=0).mean()

    print(f"Stability: {stability}, PredVar: {pred_var}")
    return stability, pred_var


@torch.no_grad()
def compute_sharpness(model, dataloader, criterion, device, rho):
    
    original_weights = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    sharpness_sum = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Perturba i pesi del modello
        for n, p in model.named_parameters():
            if p.requires_grad:
                noise = rho * torch.randn_like(p)
                p.add_(noise)

        # Calcola la perdita con i pesi perturbati
        outputs = model(inputs)
        perturbed_loss = criterion(outputs, targets)

        # Ripristina i pesi originali
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(original_weights[n])

        sharpness_sum += perturbed_loss.item()
        num_batches += 1

    sharpness=sharpness_sum / num_batches if num_batches > 0 else 0.0
    print(f"Sharpness: {sharpness}")
    return sharpness


def load_model(model_fn, model_path, device):
    model = model_fn()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_model_for_lambda(lambda_val, rho, optimizer, models_dir):
    if optimizer == "sam":
        model_path = os.path.join(models_dir, f"model_sam_lambda_{lambda_val:.2f}_rho_{rho:.2f}.pth")
    elif optimizer == "sgd":
        model_path = os.path.join(models_dir, f"model_sgd_lambda_{lambda_val:.2f}_rho_None.pth")
    else:
        raise ValueError(f"Optimizer {optimizer} non supportato.")

    if not os.path.exists(model_path):
        print(f"Modello non trovato per lambda={lambda_val:.2f} e optimizer={optimizer}")
        return None

    print(f"Caricamento modello da {model_path}")
    model = torch.load(model_path)
    return model



def plot_metric_trend(values, title, ylabel, output_file):
    
    if not values:
        print(f"[ERRORE] Nessun dato disponibile per il grafico: {title}")
        return
    lambdas, metrics = zip(*values)  # Estrae i valori di lambda e le metriche
    plt.figure()
    plt.plot(lambdas, metrics, marker='o')
    plt.title(title)
    plt.xlabel("Lambda")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Grafico salvato in {output_file}")


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

