import optuna
import torch
import torch.nn.functional as F
from torch.optim import SGD
from optimizer.optimizer import SAM


def objective(trial, model_fn, dataset, device, base_optimizer, learning_rate, momentum, weight_decay, epochs):
    
    # Prova un valore di rho
    rho = trial.suggest_float("rho", 0.01, 0.2, step=0.01)

    # Inizializza il modello e l'ottimizzatore
    model = model_fn().to(device)
    optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer, step_size=30, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, targets in dataset["train"]:
            inputs, targets = inputs.to(device), targets.to(device)

            def closure():
                loss = F.cross_entropy(model(inputs), targets)
                loss.backward()
                return loss

            optimizer.step(closure)
        scheduler.step()

    # Valutazione sul set di validazione
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataset["val"]:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += F.cross_entropy(outputs, targets, reduction="sum").item()
    val_loss /= len(dataset["val"].dataset)

    return val_loss


def optimize_rho_with_optuna(model_fn, dataset, device, base_optimizer, learning_rate, momentum, weight_decay, epochs, n_trials=20):
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_fn, dataset, device, base_optimizer, learning_rate, momentum, weight_decay, epochs), n_trials=n_trials)

    print(f"Best rho: {study.best_params['rho']} with validation loss: {study.best_value:.4f}")
    return study.best_params['rho']# grid_search_sam.py

def grid_search_sam_rho(model_fn, dataset, args, rhos=[0.01, 0.03, 0.05, 0.1], save_results=True, train_fn=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    results_file = "grid_search_sam_results.csv" if save_results else None
    if save_results:
        with open(results_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rho", "val_accuracy", "test_accuracy", "lambda_used"])

    for rho in rhos:
        print(f"\n>>> Training SAM with rho = {rho}")
        model = model_fn(num_classes=10).to(device)

        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
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

        if train_fn:
            train_fn(model, optimizer, scheduler, dataset, args, log, use_sam=True)
        else:
            raise ValueError("train_fn must be provided for grid search.")

        val_acc = log.best_metrics["val_accuracy"]
        test_acc = log.best_metrics["test_accuracy"]
        results.append((rho, val_acc, test_acc))

        print(f"Rho: {rho:.3f}, Validation Accuracy: {val_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")

        if save_results:
            with open(results_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rho, val_acc, test_acc, args.lambda_])

    best_rho, best_val_acc, best_test_acc = max(results, key=lambda x: x[1])
    print(f"\n[GRID SEARCH] Best rho: {best_rho:.3f} with Validation Accuracy: {best_val_acc*100:.2f}% and Test Accuracy: {best_test_acc*100:.2f}%")
    return best_rho, best_val_acc
