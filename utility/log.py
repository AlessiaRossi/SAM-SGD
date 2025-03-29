import time
import os
import csv
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


class Log:
    def __init__(self, log_each: int, initial_epoch=-1, log_dir="results", log_file="training_log.csv", model_name="model.pth"):
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.best_model_path = os.path.join(log_dir, model_name)
        self.model_name = model_name
        self.log_data = []

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy", "learning_rate", "elapsed", "f1_score"])

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0, "y_true": [], "y_pred": []}
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None, y_true=None, y_pred=None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate, y_true, y_pred)
        else:
            self._eval_step(loss, accuracy, y_true, y_pred)
            self._save_if_best(model)

    def flush(self) -> None:
        loss = self.epoch_state["loss"] / self.epoch_state["steps"]
        accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

        y_true = self.epoch_state["y_true"]
        y_pred = self.epoch_state["y_pred"]
        f1 = f1_score(y_true, y_pred, average='macro') if y_true and y_pred else 0.0

        if self.is_train:
            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )
        else:
            print(f"{loss:12.4f}  │{100*accuracy:10.2f} %  ┃", flush=True)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.epoch, f"{loss:.4f}", f"{accuracy:.4f}", f"{self.learning_rate:.3e}" if self.is_train else "-", self._time(), f"{f1:.4f}"])

        if not self.is_train:
            self.log_data.append((self.epoch, loss, accuracy, f1))
            self._plot_metrics()
            self._plot_comparison()

    def _train_step(self, model, loss, accuracy, learning_rate: float, y_true=None, y_pred=None) -> None:
        self.learning_rate = learning_rate
        batch_size = accuracy.size(0)
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += batch_size
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_size
        if y_true is not None and y_pred is not None:
            self.epoch_state["y_true"].extend(y_true.cpu().tolist())
            self.epoch_state["y_pred"].extend(y_pred.cpu().tolist())
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]
            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0, "y_true": [], "y_pred": []}

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

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

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")

    def _plot_metrics(self):
        if not self.log_data:
            return

        epochs, losses, accuracies, f1s = zip(*self.log_data)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, label='Loss')
        plt.plot(epochs, accuracies, label='Accuracy')
        plt.plot(epochs, f1s, label='F1-score')
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
            sam_path = os.path.join(self.log_dir, "training_sam.csv")
            sgd_path = os.path.join(self.log_dir, "training_sgd.csv")

            if os.path.exists(sam_path) and os.path.exists(sgd_path):
                sam_df = pd.read_csv(sam_path)
                sgd_df = pd.read_csv(sgd_path)

                plt.figure(figsize=(10, 5))
                plt.plot(sam_df["epoch"], sam_df["accuracy"], label="SAM", linestyle='--')
                plt.plot(sgd_df["epoch"], sgd_df["accuracy"], label="SGD", linestyle='-')
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("SGD vs SAM Accuracy")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, "comparison_accuracy.png"))
                plt.close()

                plt.figure(figsize=(10, 5))
                plt.plot(sam_df["epoch"], sam_df["loss"], label="SAM", linestyle='--')
                plt.plot(sgd_df["epoch"], sgd_df["loss"], label="SGD", linestyle='-')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("SGD vs SAM Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, "comparison_loss.png"))
                plt.close()
        except Exception as e:
            print(f"[PlotComparisonError] {e}")
