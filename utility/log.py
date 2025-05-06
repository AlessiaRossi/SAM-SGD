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
    def __init__(self, log_each, model_name, lambda_value, optimize_lambda, use_sam=False, rho=None):
        self.log_each = log_each
        self.model_name = model_name
        self.lambda_value = lambda_value
        self.optimize_lambda = optimize_lambda
        self.use_sam = use_sam
        self.rho = rho
        self.tracker = MetricsTracker()

        # Costruisci il percorso del file in base all'ottimizzatore
        if self.use_sam:
            if self.rho is None:
                raise ValueError("Il parametro 'rho' deve essere specificato per l'ottimizzatore SAM.")
            self.best_model_path = os.path.normpath(os.path.join("results", f"model_sam_lambda_{self.lambda_value:.2f}_rho_{self.rho:.2f}.pth"))
        else:
            self.best_model_path = os.path.normpath(os.path.join("results", f"model_sgd_lambda_{self.lambda_value:.2f}_rho_None.pth"))

        self.best_metrics = {
            "epoch": -1,
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "test_loss": None,
            "test_accuracy": None,
            "ece": None,
            "robust_accuracy": None,
            "flat_minima": None,
            "trades_kl": None
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
            

    def flush(self, model=None, current_accuracy=None):
        if not hasattr(self, "tracker") or self.is_train or self.tracker.steps == 0:
            return

        metrics = self.tracker.compute()
        if self.eval_label == "Validation" and metrics["accuracy"] > self.best_metrics["val_accuracy"]:
            self.best_metrics.update({"epoch": self.epoch, "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
        elif self.eval_label == "Test":
            self.best_metrics.update({"test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"]})


    def print_best_metrics(self):
        if self.best_metrics.get("epoch", -1) == -1:
            print("Nessuna metrica migliore trovata. Assicurati che le metriche vengano aggiornate durante il training.")
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
            print(f"Flat Minima: {self.best_metrics['flat_minima']:.4f}")
            print()

        if self.best_metrics["trades_kl"] is not None:
            print(f"TRADES KL Divergence: {self.best_metrics['trades_kl']:.6f}")
            print()
            
    def _save_if_best(self, model, current_accuracy):
        if self.best_model_path is not None and current_accuracy > self.best_metrics["val_accuracy"]:
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(model.state_dict(), self.best_model_path)
            self.best_metrics["val_accuracy"] = current_accuracy  # Aggiorna l'accuratezza migliore
        
    def store_trades_kl(self, avg_kl):
        self.best_metrics["trades_kl"] = avg_kl


