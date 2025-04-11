import time
import os
import csv
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


class Log:
    def __init__(self, log_each: int, initial_epoch=-1, log_dir="results", log_file="training_log.csv", model_name="model.pth", algorithm_name="", lambda_value=None, optimize_lambda=False):
        self.best_metrics = {
            "epoch": -1,
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
            "test_loss": float("inf"),
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
        }
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.best_model_path = os.path.join(log_dir, model_name)
        self.model_name = model_name
        self.algorithm_name = algorithm_name
        self.lambda_value = lambda_value
        self.optimize_lambda = optimize_lambda

        os.makedirs(self.log_dir, exist_ok=True)

        # Inizializza il file CSV con l'intestazione
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1",
                "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"
            ])

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        self.is_train = True
        self._reset(len_dataset)

    def eval(self, len_dataset: int, label="Validation") -> None:
        self.flush()
        self.is_train = False
        self.eval_label = label
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None, y_true=None, y_pred=None) -> None:
        self._eval_step(loss, accuracy, y_true, y_pred)
        if self.is_train:
            self.learning_rate = learning_rate
            self._save_if_best(model)

    def flush(self) -> None:
        if not hasattr(self, "epoch_state") or self.is_train or self.epoch_state["steps"] == 0:
            return

        loss = self.epoch_state["loss"] / self.epoch_state["steps"]
        accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]
        y_true = self.epoch_state["y_true"]
        y_pred = self.epoch_state["y_pred"]

        f1 = f1_score(y_true, y_pred, average='macro') if len(y_true) > 0 else 0.0
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) > 0 else 0.0
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) > 0 else 0.0

        if self.eval_label == "Validation":
            if accuracy > self.best_metrics["val_accuracy"]:
                self.best_metrics.update({
                    "epoch": self.epoch,
                    "val_loss": loss,
                    "val_accuracy": accuracy,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1,
                })

        elif self.eval_label == "Test":
            self.best_metrics.update({
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            })

    def save_best_metrics(self):
        """Salva i migliori parametri in un file CSV."""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.best_metrics["epoch"],
                f"{self.best_metrics['val_loss']:.4f}",
                f"{self.best_metrics['val_accuracy']:.4f}",
                f"{self.best_metrics['val_precision']:.4f}",
                f"{self.best_metrics['val_recall']:.4f}",
                f"{self.best_metrics['val_f1']:.4f}",
                f"{self.best_metrics['test_loss']:.4f}",
                f"{self.best_metrics['test_accuracy']:.4f}",
                f"{self.best_metrics['test_precision']:.4f}",
                f"{self.best_metrics['test_recall']:.4f}",
                f"{self.best_metrics['test_f1']:.4f}",
            ])

    def print_best_metrics(self):
        """Stampa i migliori parametri in output."""
        if self.best_metrics["epoch"] == -1:
            print(">>> No best metrics available to print.")
            return

        print("\nBest Metrics:")
        print(f"Epoch: {self.best_metrics['epoch']}")
        print(f"Validation Loss: {self.best_metrics['val_loss']:.4f}")
        print(f"Validation Accuracy: {self.best_metrics['val_accuracy'] * 100:.2f}%")
        print(f"Validation Precision: {self.best_metrics['val_precision'] * 100:.2f}%")
        print(f"Validation Recall: {self.best_metrics['val_recall'] * 100:.2f}%")
        print(f"Validation F1-Score: {self.best_metrics['val_f1'] * 100:.2f}%")
        print(f"Test Loss: {self.best_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {self.best_metrics['test_accuracy'] * 100:.2f}%")
        print(f"Test Precision: {self.best_metrics['test_precision'] * 100:.2f}%")
        print(f"Test Recall: {self.best_metrics['test_recall'] * 100:.2f}%")
        print(f"Test F1-Score: {self.best_metrics['test_f1'] * 100:.2f}%")
        print(f"Lambda: {self.lambda_value}")

    def _eval_step(self, loss, accuracy, y_true=None, y_pred=None) -> None:
        batch_size = accuracy.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_size
        if y_true is not None and y_pred is not None:
            self.epoch_state["y_true"].extend(y_true.cpu().tolist())
            self.epoch_state["y_pred"].extend(y_pred.cpu().tolist())

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0, "y_true": [], "y_pred": []}

