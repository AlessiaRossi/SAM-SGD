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
from optimizer.optimizer import SAM, StandardSGD
from utility.loss import CombinedLoss, FocalLoss, LogitNormLoss, TRADESLoss, HuberLoss
from utility.metrics import eval_robustness, evaluate_trades_robustness
import subprocess
import os
from weight_analysis import compute_sparsity_and_pathnorm, compute_stability_score, plot_metric_trend, load_model_for_lambda, compute_sharpness
from optimizer.rho import optimize_rho_with_optuna

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
                loss = criterion.loss1(outputs, targets)
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
    return average_loss, accuracy

def train(model, optimizer, scheduler, dataset, args, log, use_sam=False, lambda_optimizer=None, lambda_value=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_value = lambda_optimizer.current_lambda if lambda_optimizer else args.lambda_
    ce_loss = nn.CrossEntropyLoss()

    # Configura la loss in base all'argomento fornito
    if args.loss_type == "focal":
        focal_loss = FocalLoss(gamma=2, alpha=0.25)
        criterion = CombinedLoss(loss1=ce_loss, loss2=focal_loss, lambda_=lambda_value)
    elif args.loss_type == "logitnorm":
        logitnorm_loss = LogitNormLoss(device=device, t=1.0)
        criterion = CombinedLoss(loss1=ce_loss, loss2=logitnorm_loss, lambda_=lambda_value)
    elif args.loss_type == "trades":
        trades_loss = TRADESLoss(model=model, optimizer=optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf')
        criterion = CombinedLoss(loss1=ce_loss, loss2=trades_loss, lambda_=lambda_value)
    elif args.loss_type == "huber":
        huber_loss = HuberLoss(delta=1.0)
        criterion = CombinedLoss(loss1=ce_loss, loss2=huber_loss, lambda_=lambda_value)
    else:
        raise ValueError(f"Loss type '{args.loss_type}' non supportata.")
    

    best_val_accuracy = 0.0  # Per tracciare il modello migliore
    best_model_state = None  # Per salvare lo stato del modello migliore

    # Ciclo di training per il numero di epoche specificato
    for epoch in range(args.epochs):
        _train_epoch(model, dataset["train"], optimizer, scheduler, criterion, device, log, use_sam)
        
        log.eval(len_dataset=len(dataset["val"]), label="Validation")
        val_loss, val_acc = _evaluate(model, dataset["val"], criterion.loss1 if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss) else criterion, device)
        log(model, loss=val_loss, accuracy=val_acc)
        log.flush(model)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()
            
    return criterion
            
    # Salva il modello migliore alla fine del training
    if args.optimizer == "sam":
        model_path = os.path.join("results", f"model_sam_lambda_{lambda_value:.2f}_rho_{args.rho:.2f}.pth")
    else:
        model_path = os.path.join("results", f"model_sgd_lambda_{lambda_value:.2f}_rho_None.pth")

    print(f">>> Salvando il modello migliore in {model_path}")
    torch.save(best_model_state, model_path)
    log.best_model_path = model_path
    
    
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
    args = parser.parse_args()
    sparsity_values = []
    path_norm_values = []
    stability_values = []
    pred_var_values = []

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    dataset, num_classes = load_dataset(args.dataset, args.batch_size)
    model_fn = {2: WRN56_2, 4: WRN56_4, 8: WRN56_8}.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    if args.optimize_lambda:
        start, end, step = map(float, args.lambda_range.split(","))
        lambda_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]
    
    print(f">>> Training with {args.optimizer.upper()}\n>>> Loss configuration: {args.loss_type}")
    for lambda_val in lambda_values:
        model = model_fn(num_classes=num_classes).to(device)
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            use_sam = False 
            best_rho = 0
        elif args.optimizer == "sam":
            if args.optimize_rho:
                best_rho = optimize_rho_with_optuna(
                    model_fn=model_fn,
                    dataset=dataset,
                    device=device,
                    base_optimizer=torch.optim.SGD,
                    learning_rate=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    n_trials=args.n_trials,
                )
                print(f">>> Miglior rho trovato per lambda = {lambda_val:.2f}: {best_rho:.4f}")
            else:
                best_rho = args.rho
            optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD, rho=best_rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            use_sam = True

        scheduler = StepLR(optimizer.base_optimizer if args.optimizer == "sam" else optimizer, step_size=30, gamma=0.1)
        log = Log(log_each=10, model_name=f"model_{args.optimizer.lower()}_{lambda_val:.2f}.pth", lambda_value=lambda_val, optimize_lambda=args.optimize_lambda)

        # Esegui il training
        criterion = train(model, optimizer, scheduler, dataset, args, log, use_sam=use_sam, lambda_value=lambda_val)

        # Ricarica il modello migliore
        model.load_state_dict(torch.load(log.best_model_path))
        model.to(device)
        model.eval()
        print(f">>> Modello migliore ricaricato da {log.best_model_path}")
        print()


        # Calcola nuovamente validation loss e accuracy per memorizzarle nei log
        log.eval(len_dataset=len(dataset["val"]), label="Validation")
        val_loss, val_acc = _evaluate(model, dataset["val"], criterion, device)
        log(model, loss=val_loss, accuracy=val_acc)
        log.flush()

        # Test set
        test_loss, test_acc = _evaluate(model, dataset["test"], nn.CrossEntropyLoss(), device)
        log.best_metrics.update({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        # Robustezza (ECE, robust_accuracy, flat_minima)
        robustness = eval_robustness(model, dataset["test"], device, nn.CrossEntropyLoss())
        log.best_metrics.update({
            "robust_accuracy": robustness.get("robust_accuracy", 0.0),
            "flat_minima": robustness.get("flat_minima", 0.0),
            "ece": robustness.get("ece", 0.0),
        })
        # Calcola la sharpness
        sharpness = compute_sharpness(model, dataset["test"], nn.CrossEntropyLoss(), device, rho=best_rho)
        print(f"Sharpness: {sharpness:.6f}")

        # Calcola la sparsity e il path norm
        sparsity, path_norm = compute_sparsity_and_pathnorm(model)
        print(f"Sparsity: {sparsity:.6f}, Path Norm: {path_norm:.6f}")
        sparsity_values.append((lambda_val, sparsity))
        path_norm_values.append((lambda_val, path_norm))
        
        # Calcola la stability e la prediction variance
        stability, pred_var = compute_stability_score(model, dataset["test"], N=5)
        #print(f"Stability: {stability:.6f}, Prediction Variance: {pred_var:.6f}")
        stability_values.append((lambda_val, stability))
        pred_var_values.append((lambda_val, pred_var))
        
        # Valutazione specifica per TRADES
        if isinstance(criterion, CombinedLoss) and isinstance(criterion.loss2, TRADESLoss):
            evaluate_trades_robustness(model, dataset["test"], criterion, device, log)

        
        # Stampa le metriche
        log.print_best_metrics()

    print("Plotting weight analysis...")
         # Plotta le metriche
    plot_metric_trend(sparsity_values, "Sparsity vs Lambda", "Sparsity", "sparsity_vs_lambda.png")
    plot_metric_trend(path_norm_values, "PathNorm vs Lambda", "PathNorm", "pathnorm_vs_lambda.png")
    plot_metric_trend(stability_values, "Stability vs Lambda", "Stability", "stability_vs_lambda.png")
    plot_metric_trend(pred_var_values, "Prediction Variance vs Lambda", "Prediction Variance", "predvar_vs_lambda.png")
