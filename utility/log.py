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
    def __init__(self, log_each: int, initial_epoch=-1, log_dir="results", log_file="training_log.csv", model_name="model.pth", algorithm_name=""):
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.best_model_path = os.path.join(log_dir, model_name)
        self.model_name = model_name
        self.log_data = []
        self.algorithm_name = algorithm_name
        self.val_metrics = {}
        self.test_metrics = {}

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss", "val_accuracy", "test_loss", "test_accuracy"])

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
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
            self.val_metrics = {"loss": loss, "accuracy": accuracy}
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

        elif self.eval_label == "Test":
            self.test_metrics = {"loss": loss, "accuracy": accuracy}

        if hasattr(self, "val_metrics") and hasattr(self, "test_metrics"):
            val_loss = self.val_metrics.get("loss") if hasattr(self, "val_metrics") else None
            val_acc = self.val_metrics.get("accuracy") if hasattr(self, "val_metrics") else None
            test_loss = self.test_metrics.get("loss", 0.0)
            test_acc = self.test_metrics.get("accuracy", 0.0)

            if val_loss is None or val_acc is None:
                print(f"[WARNING] Validation metrics not available before test flush at epoch {self.epoch}.")
                return

            print(
                f"┃{self.epoch:12d}  ┃{val_loss:12.4f}  │{100*val_acc:10.2f} %  ┃"
                f"{test_loss:12.4f} │{100*test_acc:10.2f} %┃",
                flush=True
            )

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.epoch,
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{test_loss:.4f}",
                f"{test_acc:.4f}"
            ])



    def _eval_step(self, loss, accuracy, y_true=None, y_pred=None) -> None:
        batch_size = accuracy.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_size
        if y_true is not None and y_pred is not None:
            self.epoch_state["y_true"].extend(y_true.cpu().tolist())
            self.epoch_state["y_pred"].extend(y_pred.cpu().tolist())

    def _save_if_best(self, model):
        y_true = self.epoch_state["y_true"]
        y_pred = self.epoch_state["y_pred"]
        f1 = f1_score(y_true, y_pred, average='macro') if y_true and y_pred else 0.0
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(model.state_dict(), self.best_model_path)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0, "y_true": [], "y_pred": []}

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┳━━━━━━━╸T-E-S-T╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃             ╷         ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃       loss  │accuracy ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂─────────────┼─────────┨")

    def _plot_metrics(self):
        if not self.log_data:
            return

        epochs, losses, accuracies, f1s, precisions, recalls = zip(*self.log_data)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, label='Loss')
        plt.plot(epochs, accuracies, label='Accuracy')
        plt.plot(epochs, f1s, label='F1-score')
        plt.plot(epochs, precisions, label='Precision')
        plt.plot(epochs, recalls, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        filename = os.path.splitext(os.path.basename(self.model_name))[0]
        plt.savefig(os.path.join(self.log_dir, f"metrics_plot_{filename}.png"))
        plt.close()
        
    def _plot_comparison(self):
        try:
            import pandas as pd
            sam_path = os.path.join(self.log_dir, "evaluation_sam.csv")
            sgd_path = os.path.join(self.log_dir, "evaluation_sgd.csv")

            if os.path.exists(sam_path) and os.path.exists(sgd_path):
                sam_df = pd.read_csv(sam_path)
                sgd_df = pd.read_csv(sgd_path)

                plt.figure(figsize=(10, 5))
                plt.plot(sam_df["epoch"], sam_df["test_accuracy"], label="SAM", linestyle='--')
                plt.plot(sgd_df["epoch"], sgd_df["test_accuracy"], label="SGD", linestyle='-')
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("SGD vs SAM - Test Accuracy")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, "comparison_test_accuracy.png"))
                plt.close()

                plt.figure(figsize=(10, 5))
                plt.plot(sam_df["epoch"], sam_df["test_loss"], label="SAM", linestyle='--')
                plt.plot(sgd_df["epoch"], sgd_df["test_loss"], label="SGD", linestyle='-')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("SGD vs SAM Test Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, "comparison_test_loss.png"))
                plt.close()
        except Exception as e:
            print(f"[PlotComparisonError] {e}")
            
    def _time(self) -> str:
        elapsed_seconds = int(time.time() - self.start_time)
        return f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d} min"

