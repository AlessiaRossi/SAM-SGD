# log.py
import os
import time
import csv
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.accuracy = 0.0
        self.steps = 0
        self.y_true = []
        self.y_pred = []

    def accumulate(self, batch_loss, batch_acc, y_true=None, y_pred=None):
        batch_size = batch_acc.size(0)
        self.loss += batch_loss.sum().item()
        self.accuracy += batch_acc.sum().item()
        self.steps += batch_size
        if y_true is not None and y_pred is not None:
            self.y_true.extend(y_true.cpu().tolist())
            self.y_pred.extend(y_pred.cpu().tolist())

    def compute(self):
        avg_loss = self.loss / self.steps
        avg_acc = self.accuracy / self.steps
        if len(self.y_true) > 0:
            f1 = f1_score(self.y_true, self.y_pred, average='macro')
            precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
            recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        else:
            f1 = precision = recall = 0.0
        return avg_loss, avg_acc, precision, recall, f1


class Log:
    def __init__(self, log_each: int, initial_epoch=-1, log_dir="results", log_file="training_log.csv", model_name="model.pth", algorithm_name="", lambda_value=None, optimize_lambda=False):
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.best_model_path = os.path.join(log_dir, model_name)
        self.algorithm_name = algorithm_name
        self.lambda_value = lambda_value
        self.optimize_lambda = optimize_lambda

        self.best_metrics = {"epoch": -1, "val_loss": float("inf"), "val_accuracy": 0.0, "val_precision": 0.0, "val_recall": 0.0, "val_f1": 0.0, "test_loss": float("inf"), "test_accuracy": 0.0, "test_precision": 0.0, "test_recall": 0.0, "test_f1": 0.0}

        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"])

    def train(self, len_dataset):
        self.epoch += 1
        self.is_train = True
        self.tracker = MetricsTracker()

    def eval(self, len_dataset, label="Validation"):
        self.flush()
        self.is_train = False
        self.eval_label = label
        self.tracker = MetricsTracker()

    def __call__(self, model, loss, accuracy, learning_rate=None, y_true=None, y_pred=None):
        self.tracker.accumulate(loss, accuracy, y_true, y_pred)
        if self.is_train:
            self.learning_rate = learning_rate
            self._save_if_best(model)

    def flush(self):
        if not hasattr(self, "tracker") or self.is_train or self.tracker.steps == 0:
            return

        loss, acc, precision, recall, f1 = self.tracker.compute()
        if self.eval_label == "Validation" and acc > self.best_metrics["val_accuracy"]:
            self.best_metrics.update({"epoch": self.epoch, "val_loss": loss, "val_accuracy": acc, "val_precision": precision, "val_recall": recall, "val_f1": f1})
        elif self.eval_label == "Test":
            self.best_metrics.update({"test_loss": loss, "test_accuracy": acc, "test_precision": precision, "test_recall": recall, "test_f1": f1})

    def save_best_metrics(self):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.best_metrics["epoch"], f"{self.best_metrics['val_loss']:.4f}", f"{self.best_metrics['val_accuracy']:.4f}", f"{self.best_metrics['val_precision']:.4f}", f"{self.best_metrics['val_recall']:.4f}", f"{self.best_metrics['val_f1']:.4f}", f"{self.best_metrics['test_loss']:.4f}", f"{self.best_metrics['test_accuracy']:.4f}", f"{self.best_metrics['test_precision']:.4f}", f"{self.best_metrics['test_recall']:.4f}", f"{self.best_metrics['test_f1']:.4f}",
            ])

    def print_best_metrics(self):
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


    def _save_if_best(self, model):
        torch.save(model.state_dict(), self.best_model_path)
