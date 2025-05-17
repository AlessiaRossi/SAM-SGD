import torch
import torch.nn.functional as F


class MetricsTracker:
    def __init__(self):
        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_samples = 0
        self.steps = 0

    def reset(self):
        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_samples = 0
        self.steps = 0

    def accumulate(self, batch_loss, batch_acc):
        if isinstance(batch_acc, float):
            batch_size = 1
        elif hasattr(batch_acc, "size"):
            batch_size = batch_acc.size(0)
        else:
            raise ValueError(f"Tipo non supportato per batch_acc: {type(batch_acc)}")

        self.total_loss += batch_loss * batch_size
        self.total_acc += batch_acc * batch_size
        self.total_samples += batch_size
        self.steps += 1

    def compute(self):
        return {
            "loss": self.total_loss / self.total_samples if self.total_samples > 0 else 0.0,
            "accuracy": self.total_acc / self.total_samples if self.total_samples > 0 else 0.0,
        }


def compute_ece(probs, targets, num_bins=15):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(probs.max(dim=1).values, bin_boundaries, right=True) - 1

    ece = 0.0
    for i in range(num_bins):
        in_bin = bin_indices == i
        if in_bin.any():
            bin_acc = (targets[in_bin] == probs[in_bin].argmax(dim=1)).float().mean()
            bin_conf = probs[in_bin].max(dim=1).values.mean()
            ece += (bin_acc - bin_conf).abs() * in_bin.float().mean()

    return ece


@torch.no_grad()
def compute_flat_minima(model, inputs, targets, criterion, epsilon=1e-3):
    original_weights = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    outputs = model(inputs)
    original_loss = criterion(outputs, targets)

    for n, p in model.named_parameters():
        if p.requires_grad:
            noise = epsilon * torch.randn_like(p)
            p.add_(noise)

    perturbed_outputs = model(inputs)
    perturbed_loss = criterion(perturbed_outputs, targets)

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(original_weights[n])

    return abs((perturbed_loss - original_loss).item())


def eval_robustness(model, dataloader, device, criterion, lambda_value=None):
    model.eval()
    robust_correct = 0
    total = 0
    ece_sum = 0.0
    flat_minima_sum = 0.0
    num_batches = 0

    criterion_to_use = criterion.loss1 if hasattr(criterion, 'loss2') and criterion.loss2.__class__.__name__ == "TRADESLoss" else criterion

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            ece_sum += compute_ece(probs, targets)

        flat_minima_sum += compute_flat_minima(model, inputs, targets, criterion_to_use)

        adv_inputs = inputs + 0.031 * torch.randn_like(inputs).sign()
        adv_inputs = torch.clamp(adv_inputs, 0.0, 1.0)

        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            preds = adv_outputs.argmax(dim=1)
            robust_correct += (preds == targets).sum().item()

        num_batches += 1

    robust_accuracy = robust_correct / total
    ece = ece_sum / num_batches if num_batches > 0 else 0.0
    flat_minima = flat_minima_sum / num_batches if num_batches > 0 else 0.0

    return {
        "robust_accuracy": robust_accuracy,
        "flat_minima": flat_minima,
        "ece": ece,
    }


def evaluate_trades_robustness(model, dataloader, criterion, device, log):
    if not (isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss)):
        print("Criterion non compatibile con TRADES. Salto la valutazione di robustezza.")
        return

    print(">>> Inizio valutazione TRADES robustness...")
    kl_div = criterion.loss2.kl_div
    model.eval()
    kl_values = []

    with torch.no_grad():
        for x, targets in dataloader:
            x, targets = x.to(device), targets.to(device)
            x_adv = criterion.loss2.generate_adversarial(x)  # Genera perturbazioni avversarie

            # Debug: Verifica la generazione di x_adv
            print(f"Debug: x shape = {x.shape}, x_adv shape = {x_adv.shape}")
            print(f"Debug: x_adv min = {x_adv.min().item()}, max = {x_adv.max().item()}")

            logits_clean = model(x)
            logits_adv = model(x_adv)
            kl_value = kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean, dim=1))
            kl_values.append(kl_value.item())

    # Calcola la media dei valori KL
    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    log.best_metrics.update({"trades_kl": avg_kl})
    print(f">>> TRADES KL Divergence: {avg_kl:.6f}")


import csv

def read_metrics_csv(filepath):
    """
    Legge il file metrics_summary.csv e restituisce i dati come una lista di dizionari.

    Args:
        filepath (str): Percorso del file CSV.

    Returns:
        list[dict]: Lista di dizionari con le metriche.
    """
    metrics = []
    with open(filepath, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Converti i valori numerici in float, se possibile
            metrics.append({key: float(value) if value.replace('.', '', 1).isdigit() else value for key, value in row.items()})
    return metrics