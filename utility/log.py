# utility/log.py (semplificato + stampa metriche robuste)
import os
import torch
from utility.metrics import MetricsTracker, eval_robustness


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()

def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


class Log:
    def __init__(self, log_each, model_name, lambda_value, optimize_lambda, use_sam=False):
        self.log_each = log_each
        self.model_name = model_name
        self.lambda_value = lambda_value
        self.optimize_lambda = optimize_lambda
        self.use_sam = use_sam
        self.metrics = {
            "train": [],
            "val": [],
            "test": [],
            "robustness": [],
        }
        self.best_model_path = f"results/{model_name}"
        # Inizializza tutte le chiavi necessarie in best_metrics
        self.best_metrics = {
            "epoch": -1,
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "test_loss": None,
            "test_accuracy": None,
            "sharpness": 0.0,  # Inizializza con un valore predefinito
            "flat_minima": 0.0,
            "robust_accuracy": 0.0,
            "ece": 0.0,
        }
        self.epoch = 0

    def train(self, len_dataset):
        self.epoch += 1
        self.is_train = True
        self.tracker = MetricsTracker()

    def eval(self, len_dataset, label="Validation"):
        self.flush()
        self.is_train = False
        self.eval_label = label
        self.tracker = MetricsTracker()

    def __call__(self, model, loss, accuracy, learning_rate=None, **_):
        self.tracker.accumulate(loss, accuracy)
        if self.is_train:
            self.learning_rate = learning_rate
            self._save_if_best(model)

    def flush(self):
        if not hasattr(self, "tracker") or self.is_train or self.tracker.steps == 0:
            return

        metrics = self.tracker.compute()
        if self.eval_label == "Validation" and metrics["accuracy"] > self.best_metrics["val_accuracy"]:
            self.best_metrics.update({"epoch": self.epoch, "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
        elif self.eval_label == "Test":
            self.best_metrics.update({"test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"]})

    def save_best_metrics(self):
        pass

    def print_best_metrics(self):
        if "epoch" not in self.best_metrics or self.best_metrics["epoch"] == -1:
            print("Nessuna metrica migliore trovata durante il training.")
            return

        print("\nBest Metrics:")
        print(f"Epoch: {self.best_metrics['epoch']}")
        print(f"Validation Loss: {self.best_metrics['val_loss']:.4f}")
        print(f"Validation Accuracy: {self.best_metrics['val_accuracy'] * 100:.2f}%")
        print(f"Test Loss: {self.best_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {self.best_metrics['test_accuracy'] * 100:.2f}%")
        print(f"Lambda: {self.lambda_value}")
        print(f"Expected Calibration Error (ECE): {self.best_metrics['ece']:.4f}") 

        if self.best_metrics["robust_accuracy"] is not None:
            print("[Robustness Metrics]")
            print(f"Robust Accuracy: {self.best_metrics['robust_accuracy'] * 100:.2f}%")
            print(f"Sharpness: {self.best_metrics['sharpness']:.4f}")
            print(f"Flat Minima: {self.best_metrics['flat_minima']:.4f}")

    def _save_if_best(self, model):
        torch.save(model.state_dict(), self.best_model_path)

    def attach_robust_metrics(self, model, dataloader, device, criterion):
        """
        Calcola e salva le metriche di robustezza.
        """
        robustness = eval_robustness(model, dataloader, device, criterion, lambda_value=self.lambda_value)
        self.best_metrics.update({
            "robust_accuracy": robustness.get("robust_accuracy", 0.0),
            "sharpness": robustness.get("sharpness", 0.0),
            "flat_minima": robustness.get("flat_minima", 0.0),
            "ece": robustness.get("ece", 0.0),
        })
