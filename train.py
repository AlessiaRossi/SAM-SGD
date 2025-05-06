import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from model.Net import WRN56_2, WRN56_4, WRN56_8
from utility.log import Log
from utility.initialize import initialize
from utility.bypass_bn import enable_running_stats, disable_running_stats
from optimizer.optimizer import SAM, StandardSGD, create_optimizer
from utility.loss import CombinedLoss, FocalLoss, LogitNormLoss, TRADESLoss, HuberLoss, SALoss
from utility.metrics import eval_robustness, evaluate_trades_robustness
import subprocess
import os
import csv
from weight_analysis import compute_sparsity_and_pathnorm, compute_stability_score, plot_metric_trend, load_model_for_lambda, compute_sharpness
from optimizer.rho import optimize_rho_with_optuna
import optuna

def save_metrics_csv(metrics_summary, config, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_summary.csv")

    # Verifica se il file esiste per scrivere l'header solo una volta
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as csv_file:  # Usa modalità "a" per appendere
        fieldnames = ["lambda"] + list(metrics_summary.keys()) + list(config.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Scrivi l'header solo se il file non esiste

        # Combina le metriche con la configurazione
        row = {**metrics_summary, **config}
        writer.writerow(row)

    print(f"Risultati aggiunti a {csv_path}")
        
def _train_epoch(model, dataloader, optimizer, scheduler, criterion, device, log, use_sam):
    model.train()
    log.train(len_dataset=len(dataloader))
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Ensure inputs require gradients
        inputs.requires_grad_()

        # Ensure targets are 1D
        if targets.dim() != 1:
            targets = targets.argmax(dim=1)
            
        if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss):
            # CASO SPECIALE: se loss2 è TRADESLoss
            loss = criterion.loss2(model, inputs,targets)  
            # Usa TRADES per il training, loss2 gestisce tutto (crea anche gli adversarial examples)

        if use_sam:
            def closure():
                loss = criterion(model, inputs, targets)  
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = criterion(model, inputs, targets) 
            loss.backward()
            optimizer.step()

    scheduler.step()


def _evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Ottieni le predizioni dal modello

            if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss):
                # Evita perturbazioni adversariali in validazione: usa solo la CE
                loss = criterion(model, inputs, targets)
            elif isinstance(criterion, CombinedLoss):
                loss = criterion(model, inputs, targets)
            else:
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)  # Predizioni finali
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Valutazione - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    return average_loss, accuracy

def train(model, optimizer, scheduler, dataset, args, log, use_sam=False, lambda_optimizer=None, lambda_value=None, model_path=None):
    """
    Funzione per allenare il modello e salvare il modello migliore.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_value = lambda_optimizer.current_lambda if lambda_optimizer else args.lambda_
    ce_loss = nn.CrossEntropyLoss()

    # Configura la loss in base all'argomento fornito
    if args.loss_type == "focal":
        focal_loss = FocalLoss(gamma=2, alpha=0.25)
        criterion = CombinedLoss(loss1=ce_loss, loss2=focal_loss, lambda_=lambda_value)
    elif args.loss_type == "saloss":
        saloss = SALoss(noise_std=0.1)
        criterion = CombinedLoss(loss1=ce_loss, loss2=saloss, lambda_=lambda_value)
    elif args.loss_type == "logitnorm":
        logitnorm_loss = LogitNormLoss(device=device, t=1.0)
        criterion = CombinedLoss(loss1=ce_loss, loss2=logitnorm_loss, lambda_=lambda_value)
    elif args.loss_type == "trades":
        trades_loss = TRADESLoss(model=model, optimizer=optimizer, step_size=0.003, epsilon=0.031, perturb_steps=5, beta=1.0, distance='l_inf')
        criterion = CombinedLoss(loss1=ce_loss, loss2=trades_loss, lambda_=lambda_value)
    elif args.loss_type == "huber":
        huber_loss = HuberLoss(delta=1.0)
        criterion = CombinedLoss(loss1=ce_loss, loss2=huber_loss, lambda_=lambda_value)
    else:
        raise ValueError(f"Loss type '{args.loss_type}' non supportata.")
    
    best_val_accuracy = 0.0  # Traccia l'accuratezza migliore
    best_model_path = None   # Percorso del miglior modello

    # Variabili per tracciare il miglior modello del trial corrente
    best_val_accuracy_trial = 0.0
    best_val_loss_trial = 0.0
    best_model_path_trial = None

    # Ciclo di training per il numero di epoche specificato
    for epoch in range(args.epochs):
        _train_epoch(model, dataset["train"], optimizer, scheduler, criterion, device, log, use_sam)
        
        log.eval(len_dataset=len(dataset["val"]), label="Validation")
        val_loss, val_acc = _evaluate(model, dataset["val"], criterion.loss1 if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss) else criterion, device)
        log(model, loss=val_loss, accuracy=val_acc)
        
        # Determina il prefisso e il valore di rho in base all'ottimizzatore
        if isinstance(optimizer, SAM):
            if args.rho is None:
                raise ValueError("Il parametro 'rho' deve essere specificato per l'ottimizzatore SAM.")
            model_path = f"results/model_sam_lambda_{lambda_value:.2f}_rho_{args.rho:.2f}.pth"
        else:  # Caso SGD
            model_path = f"results/model_sgd_lambda_{lambda_value:.2f}_rho_None.pth"

        # Normalizza il percorso
        model_path = os.path.normpath(model_path)

        # Salva il modello migliore solo se supera il precedente
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = model_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

        # Aggiorna il miglior modello del trial corrente
        if val_acc > best_val_accuracy_trial:
            best_val_accuracy_trial = val_acc
            best_val_loss_trial = val_loss
            best_model_path_trial = model_path

    # Alla fine del trial, salva e stampa il miglior modello
    if best_model_path_trial:
        best_model_path_trial = os.path.normpath(best_model_path_trial)  # Normalizza il percorso
        os.makedirs(os.path.dirname(best_model_path_trial), exist_ok=True)
        torch.save(model.state_dict(), best_model_path_trial)
        print(f">>> Salvando il miglior modello del trial in {best_model_path_trial} con accuratezza: {best_val_accuracy_trial:.4f}")

    # Alla fine del trial, stampa la valutazione finale
    print(f"Valutazione finale - Loss: {best_val_loss_trial:.4f}, Accuracy: {best_val_accuracy_trial:.4f}")

    # Verifica che il file esista prima di caricarlo
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Il file del modello non esiste: {best_model_path}")

    # Ricarica il miglior modello
    print(f">>> Modello migliore ricaricato da {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    return criterion

def optimize_lambda_with_optuna(train_fn, model_fn, dataset, args, n_trials=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        lambda_value = trial.suggest_float("lambda_", 0.0, 1.0, step=0.05)
        args.lambda_ = lambda_value

        model = model_fn(num_classes=dataset["train"].dataset.dataset.classes).to(device)
        optimizer, use_sam = create_optimizer(model, args)
        scheduler = StepLR(optimizer.base_optimizer if args.optimizer == "sam" else optimizer, step_size=30, gamma=0.1)

        # Genera il nome del file per il modello
        if args.optimizer == "sam":
            model_path = os.path.join("results", f"model_sam_lambda_{lambda_value:.2f}_rho_{args.rho:.2f}.pth")
        else:
            model_path = os.path.join("results", f"model_sgd_lambda_{lambda_value:.2f}_rho_None.pth")

        # Normalizza il percorso per evitare duplicazioni
        if not model_path.startswith("results/"):
            model_path = os.path.join("results", model_path)
        model_path = os.path.normpath(model_path)  # Rimuove duplicazioni come 'results/results/'

        # Aggiungi il parametro model_name
        log = Log(
            log_each=10,
            model_name=model_path,
            lambda_value=lambda_value,
            optimize_lambda=True
        )

        criterion = train_fn(model, optimizer, scheduler, dataset, args, log, use_sam=use_sam, lambda_value=lambda_value, model_path=model_path)

        val_loss, val_acc = _evaluate(model, dataset["val"], criterion.loss1 if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss) else criterion, device)
        return val_acc 

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_lambda = study.best_params["lambda_"]
    print(f">>> Miglior Lambda trovato: {best_lambda:.4f}")

    return best_lambda

def optimize_rho_lambda_with_optuna(train_fn, model_fn, dataset, args, n_trials=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_global_val_acc = 0.0
    best_global_model_path = None
    best_params = {}
    
    def objective(trial):
        nonlocal best_global_val_acc, best_global_model_path, best_params
        
        rho = trial.suggest_float("rho", 0.01, 0.2, step=0.01)
        lambda_value = trial.suggest_float("lambda_", 0.0, 1.0, step=0.05)

        args.rho = rho
        args.lambda_ = lambda_value

        model = model_fn(num_classes=dataset["train"].dataset.dataset.classes).to(device)
        optimizer, use_sam = create_optimizer(model, args)
        scheduler = StepLR(optimizer.base_optimizer if use_sam else optimizer, step_size=30, gamma=0.1)

        model_path = f"results/model_sam_lambda_{lambda_value:.2f}_rho_{rho:.2f}.pth"
        log = Log(
            log_each=10,
            model_name=model_path,
            lambda_value=lambda_value,
            optimize_lambda=True,
            use_sam=use_sam,
            rho=rho
        )

        criterion = train_fn(model, optimizer, scheduler, dataset, args, log, use_sam=use_sam, lambda_value=lambda_value, model_path=model_path)
        val_loss, val_acc = _evaluate(model, dataset["val"], criterion.loss1 if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss) else criterion, device)
        
        print(f"[Trial {trial.number}] lambda: {lambda_value:.2f}, rho: {rho:.2f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_global_val_acc:
            best_global_val_acc = val_acc
            best_global_model_path = model_path
            best_params = {"rho": rho, "lambda_": lambda_value}
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f">>> 🔥 Nuovo miglior modello globale salvato in {best_global_model_path}")
        else:
            print(">>> ❌ Trial peggiore del migliore globale. Nessun aggiornamento del modello.")


        return val_acc  

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_rho = study.best_params["rho"]
    best_lambda = study.best_params["lambda_"]
    print(f">>> Miglior combinazione trovata: rho = {best_rho:.3f}, lambda = {best_lambda:.2f}, val_acc = {-study.best_value:.4f}")
    print(f">>> Modello migliore salvato in: {best_global_model_path}")


    return best_rho, best_lambda


def load_dataset(name, batch_size):
    data_dir = "./data"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if name.lower() == "cifar10":
        dataset_cls = datasets.CIFAR10
        num_classes = 10
    elif name.lower() == "cifar100":
        dataset_cls = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    full_train_dataset = dataset_cls(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = dataset_cls(root=data_dir, train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
    }, num_classes


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

    print("Training completo e metriche salvate!")
