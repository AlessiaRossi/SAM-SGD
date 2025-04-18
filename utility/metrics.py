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
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].float().mean().item()
            avg_confidence_in_bin = probs[in_bin].mean().item()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece


def eval_robustness(model, dataloader, device, criterion, epsilon=0.031, distance='l_inf', flat_minima_epsilon=1e-3, lambda_value=None):
    model.eval()
    robust_correct = 0
    total = 0
    sharpness_sum = 0.0
    flat_minima_sum = 0.0
    num_batches = 0
    ece_sum = 0.0

    original_state = {name: param.clone() for name, param in model.named_parameters()}

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        with torch.no_grad():
            outputs = model(inputs)
            original_loss = criterion(outputs, targets)

            # Calcola le probabilità predette
            probs = F.softmax(outputs, dim=1).max(dim=1).values
            ece_sum += compute_ece(probs, targets)

        if distance == 'l_inf':
            adv_inputs = inputs + epsilon * torch.randn_like(inputs).sign()
        elif distance == 'l_2':
            delta = torch.randn_like(inputs)
            delta = delta / delta.view(inputs.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            adv_inputs = inputs + epsilon * delta
        else:
            raise ValueError(f"Unsupported distance: {distance}")

        adv_inputs = torch.clamp(adv_inputs, 0.0, 1.0)

        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, targets)
            preds = adv_outputs.argmax(dim=1)
            robust_correct += (preds == targets).sum().item()

        sharpness_sum += abs(adv_loss - original_loss).item()

        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(flat_minima_epsilon * torch.randn_like(param))

            perturbed_outputs = model(inputs)
            perturbed_loss = criterion(perturbed_outputs, targets)

            for name, param in model.named_parameters():
                param.copy_(original_state[name])

        flat_minima_sum += abs(perturbed_loss - original_loss).item()
        num_batches += 1

    robust_accuracy = robust_correct / total
    sharpness = sharpness_sum / num_batches if num_batches > 0 else 0.0
    flat_minima = flat_minima_sum / num_batches if num_batches > 0 else 0.0
    ece = ece_sum / num_batches if num_batches > 0 else 0.0

    return {
        "robust_accuracy": robust_accuracy,
        "sharpness": sharpness,
        "flat_minima": flat_minima,
        "ece": ece,
        "lambda": lambda_value
    }
