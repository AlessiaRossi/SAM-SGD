import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utility.loss import CombinedLoss, TRADESLoss
from utility.log import Log
from utility.metrics import eval_robustness,read_metrics_csv
from utility.weight_analysis import compute_sparsity_and_pathnorm, compute_stability_score, compute_sharpness
from train import train, load_dataset, _evaluate, save_metrics_csv
from optimizer.optuna_optimization import optimize_lambda_with_optuna, optimize_rho_lambda_with_optuna
from utility.initialize import initialize
from model.Net import WRN56_2, WRN56_4, WRN56_8
from optimizer.rho import optimize_rho_with_optuna
from torch.optim.lr_scheduler import StepLR
from optimizer.optimizer import create_optimizer
from utility.visualization import plot_pareto_front, plot_spider
import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--rho", default=0.05, type=float)
    parser.add_argument("--optimize_rho", action="store_true")
    parser.add_argument("--optimize_lambda", action="store_true")
    parser.add_argument("--lambda_", default=1.0, type=float)
    parser.add_argument("--lambda_range", default="0,1,0.2", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--loss_type", default="focal", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "sam"])
    parser.add_argument("--n_trials", default=10, type=int)
    parser.add_argument("--optimize_rho_lambda", action="store_true")
    args = parser.parse_args()
    sparsity_values = []
    path_norm_values = []
    stability_values = []
    pred_var_values = []

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    dataset, num_classes = load_dataset(args.dataset, args.batch_size)
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=4)  # aggiunto 
    model_fn = {2: WRN56_2, 4: WRN56_4, 8: WRN56_8}.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    if args.optimize_rho:
        print(">>> Ottimizzazione di rho in corso...")
        best_rho = optimize_rho_with_optuna(
            model_fn=model_fn,
            dataset=dataset,
            device=device,
            base_optimizer=torch.optim.SGD,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            n_trials=args.n_trials
        )
        args.rho = best_rho
        print(f">>> Miglior rho trovato: {args.rho}")
        
    

    if args.optimize_lambda:
        best_lambda = optimize_lambda_with_optuna(train, model_fn, dataset, args, n_trials=args.n_trials)
        args.lambda_ = best_lambda
        lambda_values = [best_lambda]
    else:
        start, end, step = map(float, args.lambda_range.split(","))
        lambda_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]
    print(f">>> Training with {args.optimizer.upper()}\n>>> Loss configuration: {args.loss_type}")
    metrics_summary = []
    
    if args.optimize_rho_lambda:
        print(">>> Ottimizzazione combinata di rho e lambda in corso...")
        best_rho, best_lambda = optimize_rho_lambda_with_optuna(train, model_fn, dataset, args, n_trials=args.n_trials)
        args.rho = best_rho
        args.lambda_ = best_lambda
        lambda_values = [best_lambda]

    elif args.optimize_lambda:
        best_lambda = optimize_lambda_with_optuna(train, model_fn, dataset, args, n_trials=args.n_trials)
        args.lambda_ = best_lambda
        lambda_values = [best_lambda]

    else:
        start, end, step = map(float, args.lambda_range.split(","))
        lambda_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]

    print(f">>> Training with {args.optimizer.upper()}\n>>> Loss configuration: {args.loss_type}")
    for lambda_value in lambda_values:
        model = model_fn(num_classes=num_classes).to(device)
        optimizer, use_sam = create_optimizer(model, args)
        scheduler = StepLR(optimizer.base_optimizer if use_sam else optimizer, step_size=30, gamma=0.1)

        if args.optimizer == "sam":
            model_path = os.path.join("results", f"model_sam_lambda_{lambda_value:.2f}_rho_{args.rho:.2f}.pth")
        else:
            model_path = os.path.join("results", f"model_sgd_lambda_{lambda_value:.2f}_rho_None.pth")

        log = Log(
            log_each=10,
            model_name=model_path,
            lambda_value=lambda_value,
            optimize_lambda=args.optimize_lambda,
            use_sam=use_sam,
            rho=args.rho if use_sam else None
        )

        if os.path.exists(log.best_model_path):
            model.load_state_dict(torch.load(log.best_model_path))
            model.to(device)
            print(f">>> Modello migliore ricaricato da {log.best_model_path}")
        else:
            print(f">>> ⚠️ Nessun file trovato in {log.best_model_path}, skip del caricamento modello.")

        # Carica il modello
        model.load_state_dict(torch.load(log.best_model_path))

        # Esegui il training
        criterion = train(model, optimizer, scheduler, dataset, args, log, use_sam=use_sam, lambda_value=lambda_value, model_path=model_path)

        # Salvataggio del miglior modello
        if log.best_model_path:
            log.best_model_path = os.path.normpath(log.best_model_path)  # Normalizza il percorso
            model.load_state_dict(torch.load(log.best_model_path))
            print(f">>> Modello migliore ricaricato da {log.best_model_path}")
            print()

        # Calcola nuovamente validation loss e accuracy per memorizzarle nei log
        log.eval(len_dataset=len(dataset["val"]), label="Validation")
        val_loss, val_acc = _evaluate(
            model,
            dataset["val"],
            criterion.loss1 if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss) else criterion,
            device
        )
        log(model, loss=val_loss, accuracy=val_acc)
        log.flush(model=model, current_accuracy=val_acc)

        # Test set
        test_loss, test_acc = _evaluate(model, dataset["test"], nn.CrossEntropyLoss(), device)

        robustness = eval_robustness(model, dataset["test"], device, nn.CrossEntropyLoss())
        sharpness = compute_sharpness(model, dataset["test"], nn.CrossEntropyLoss(), device, rho=args.rho)
        sparsity, path_norm = compute_sparsity_and_pathnorm(model)
        stability, pred_var = compute_stability_score(model, dataset["test"], N=5)

        metrics_summary = {
            "lambda": lambda_value,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "sharpness": sharpness,
            "sparsity": sparsity,
            "path_norm": path_norm,
            "stability": stability,
            "prediction_variance": pred_var,
            "robust_accuracy": robustness.get("robust_accuracy", 0.0),
            "flat_minima": robustness.get("flat_minima", 0.0),
            "ece": robustness.get("ece", 0.0)
        }

        # Configurazione corrente
        config = {
            "loss_type": args.loss_type,
            "optimizer": args.optimizer,
            "rho": args.rho if args.optimizer == "sam" else None
        }
        
        print("\n>>> Metriche calcolate:")
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

        # Salva le metriche nel file CSV
        save_metrics_csv(metrics_summary, config)
    
    metrics_csv_path = "results/metrics_summary.csv"
    if os.path.exists(metrics_csv_path):
        print(f">>> Caricamento dei risultati da {metrics_csv_path}...")
        metrics_summary = read_metrics_csv(metrics_csv_path)

        # Visualizzazione dei risultati
        print("\n>>> Visualizzazione dei risultati:")
        plot_pareto_front(
            metrics_summary,
            x_metric="test_loss",
            y_metric="val_accuracy",
            title="Pareto Front: Test Loss vs Validation Accuracy",
            output_path="results/pareto_front.png"
        )

        config_labels = [f"Lambda={m['lambda']}" for m in metrics_summary]
        plot_spider(
            metrics_summary,
            config_labels,
            loss_names=["focal" , "logitnorm", "trades", "huber" , "saloss"],
            title="Comparison of Configurations",
            output_path="results/spider_plot.png"
        )
    else:
        print(f">>> ⚠️ Il file {metrics_csv_path} non esiste. Assicurati di aver salvato i risultati prima di visualizzarli.")

    print("Training completo e metriche salvate!")
