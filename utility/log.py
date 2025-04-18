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
    def __init__(self, log_each: int, initial_epoch=-1, log_dir="results", model_name="model.pth", algorithm_name="", lambda_value=None, optimize_lambda=False):
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_dir = log_dir
        self.best_model_path = os.path.join(log_dir, model_name)
        self.algorithm_name = algorithm_name
        self.lambda_value = lambda_value
        self.optimize_lambda = optimize_lambda

        self.best_metrics = {
            "epoch": -1,
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "test_loss": float("inf"),
            "test_accuracy": 0.0,
            "robust_accuracy": None,
            "sharpness": None,
            "flat_minima": None,
            "ece": None, 
        }

        os.makedirs(self.log_dir, exist_ok=True)

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
        if self.best_metrics["epoch"] == -1:
            print(">>> No best metrics available to print.")
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
        robustness = eval_robustness(model, dataloader, device, criterion, lambda_value=self.lambda_value)
        self.best_metrics.update(robustness)
