import os
import csv
import torch
import torch.nn as nn   
import torch.nn.functional as F
import numpy as np
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
from utility.weight_analysis import compute_sparsity_and_pathnorm, compute_stability_score, plot_metric_trend, load_model_for_lambda, compute_sharpness
from optimizer.rho import optimize_rho_with_optuna

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

