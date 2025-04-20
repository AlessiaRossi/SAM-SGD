# utility/metrics.py (eval_robustness aggiornato: integrabile con log.print_best_metrics)
import torch
import torch.nn.functional as F

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.accuracy = 0.0
        self.steps = 0

    def accumulate(self, batch_loss, batch_acc, **_):
        batch_size = batch_acc.size(0)
        self.loss += batch_loss.sum().item()
        self.accuracy += batch_acc.sum().item()
        self.steps += batch_size

    def compute(self):
        return {
            "loss": self.loss / self.steps,
            "accuracy": self.accuracy / self.steps,
        }


def compute_ece(probs, targets, num_bins=15):
    """
    Calcola l'Expected Calibration Error (ECE).

    Args:
        probs (torch.Tensor): Probabilità predette (output di softmax).
        targets (torch.Tensor): Target reali.
        num_bins (int): Numero di bin per la calibrazione.

    Returns:
        float: Valore dell'ECE.
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(probs.max(dim=1).values, bin_boundaries, right=True) - 1

    ece = 0.0
    for i in range(num_bins):
        in_bin = bin_indices == i
        if in_bin.any():
            bin_acc = (targets[in_bin] == probs[in_bin].argmax(dim=1)).float().mean()
            bin_conf = probs[in_bin].max(dim=1).values.mean()
            ece += (bin_acc - bin_conf).abs() * in_bin.float().mean()

    return ece.item()


@torch.no_grad()
def compute_flat_minima(model, inputs, targets, criterion, epsilon=1e-3):
    """
    Calcola la Flat Minima perturbando i pesi del modello.

    Args:
        model (torch.nn.Module): Modello.
        inputs (torch.Tensor): Input del modello.
        targets (torch.Tensor): Target reali.
        criterion (torch.nn.Module): Funzione di perdita.
        epsilon (float): Ampiezza della perturbazione.

    Returns:
        float: Valore della Flat Minima.
    """
    original_weights = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Calcola la perdita originale
    outputs = model(inputs)
    original_loss = criterion(outputs, targets)

    # Perturba i pesi
    for n, p in model.named_parameters():
        if p.requires_grad:
            noise = epsilon * torch.randn_like(p)
            p.add_(noise)

    # Calcola la perdita perturbata
    perturbed_outputs = model(inputs)
    perturbed_loss = criterion(perturbed_outputs, targets)

    # Ripristina i pesi originali
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(original_weights[n])

    # Calcola la Flat Minima come differenza tra le perdite
    return abs((perturbed_loss - original_loss).item())


def eval_robustness(model, dataloader, device, criterion, lambda_value=None):
    """
    Calcola le metriche di robustezza, inclusa la sharpness, l'ECE e la Flat Minima.
    """
    model.eval()
    robust_correct = 0
    total = 0
    sharpness_sum = 0.0
    ece_sum = 0.0
    flat_minima_sum = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        # Calcola le probabilità e l'ECE
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            ece_sum += compute_ece(probs, targets)

        # Calcola la Flat Minima
        flat_minima_sum += compute_flat_minima(model, inputs, targets, criterion)

        # Perturba gli input per calcolare la robustezza
        adv_inputs = inputs + 0.031 * torch.randn_like(inputs).sign()
        adv_inputs = torch.clamp(adv_inputs, 0.0, 1.0)

        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, targets)
            preds = adv_outputs.argmax(dim=1)
            robust_correct += (preds == targets).sum().item()

        # Calcola la sharpness come differenza tra le perdite
        sharpness_sum += abs(adv_loss - criterion(outputs, targets)).item()
        num_batches += 1

    robust_accuracy = robust_correct / total
    sharpness = sharpness_sum / num_batches if num_batches > 0 else 0.0
    ece = ece_sum / num_batches if num_batches > 0 else 0.0
    flat_minima = flat_minima_sum / num_batches if num_batches > 0 else 0.0

    return {
        "robust_accuracy": robust_accuracy,
        "sharpness": sharpness,
        "flat_minima": flat_minima,
        "ece": ece,
    }


def evaluate_trades_robustness(model, dataloader, criterion, device, log):
    """
    Valuta la robustezza del modello utilizzando TRADES.
    """
    if not (isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss)):
        print("Criterion non compatibile con TRADES. Salto la valutazione di robustezza.")
        return

    kl_div = criterion.loss2.kl_div
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Genera esempi avversari
            x_adv = criterion.loss2.generate_adversarial(x)
            # Calcola i logits per input puliti e avversari
            logits_clean = model(x)
            logits_adv = model(x_adv)
            # Calcola la KL divergence
            kl_value = kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean, dim=1))
            log.store_trades_kl(kl_value.item())
