import torch
from utility.lr import StepLR
from utility.log import Log  # Import Log
import csv  # Import csv module



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
def grid_search_sam_rho(model_fn, dataset, args, rhos=[0.01, 0.03, 0.05, 0.1], save_results=True, train_fn=None):
    """
    Esegue una ricerca a griglia sui valori di rho per l'ottimizzatore SAM.

    Args:
        model_fn: Funzione per creare il modello.
        dataset: Dataset suddiviso in train, val e test.
        args: Argomenti di configurazione.
        rhos: Lista di valori di rho da testare.
        save_results: Se True, salva i risultati in un file CSV.

    Returns:
        best_rho: Il valore di rho con la migliore accuratezza di validazione.
        best_acc: La migliore accuratezza di validazione.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # File per salvare i risultati
    results_file = "grid_search_sam_results.csv" if save_results else None
    if save_results:
        with open(results_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rho", "val_accuracy", "test_accuracy", "lambda_used"])

    for rho in rhos:
        print(f"\n>>> Training SAM with rho = {rho}")
        model = model_fn(num_classes=10).to(device)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=rho,
            adaptive=False,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer.base_optimizer, args.learning_rate, args.epochs)

        log_file = f"evaluation_sam_rho_{rho:.3f}.csv"
        model_name = f"model_sam_rho_{rho:.3f}.pth"
        log = Log(
            log_each=10,
            log_file=log_file,
            model_name=model_name,
            algorithm_name=f"SAM_rho_{rho:.3f}",
            lambda_value=args.lambda_,
            optimize_lambda=args.optimize_lambda
        )

        # Esegui il training
        train_fn(model, optimizer, scheduler, dataset, args, log, use_sam=True)


        # Salva i risultati
        val_acc = log.best_metrics["val_accuracy"]
        test_acc = log.best_metrics["test_accuracy"]
        results.append((rho, val_acc, test_acc))
        print(f"Rho: {rho:.3f}, Validation Accuracy: {val_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")

        if save_results:
            with open(results_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rho, val_acc, test_acc, args.lambda_])

    # Trova il miglior rho
    best_rho, best_val_acc, best_test_acc = max(results, key=lambda x: x[1])
    print(f"\n[GRID SEARCH] Best rho: {best_rho:.3f} with Validation Accuracy: {best_val_acc*100:.2f}% and Test Accuracy: {best_test_acc*100:.2f}%")
    return best_rho, best_val_acc
