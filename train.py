# train.py (aggiornato: esegue anche plot_weight_analysis.py alla fine)
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
from scripts.SAM import SAM
from utility.loss import CombinedLoss, FocalLoss, LogitNormLoss, TRADESLoss, HuberLoss
import subprocess


def _train_epoch(model, dataloader, optimizer, scheduler, criterion, device, log, use_sam):
    model.train()
    log.train(len_dataset=len(dataloader))
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        if use_sam:
            enable_running_stats(model)
            loss = criterion(model, inputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            loss = criterion(model, inputs, targets)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss = criterion(model, inputs, targets)
            loss.backward()
            optimizer.step()
    scheduler.step()


def _evaluate(model, dataloader, criterion, device, log, split_name):
    model.eval()
    log.eval(len_dataset=len(dataloader), label=split_name)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(model, inputs, targets)
            preds = predictions.argmax(dim=1)
            accuracy = (preds == targets).float().mean()
            acc_tensor = accuracy.repeat(targets.size(0))
            log(model, loss.cpu(), acc_tensor.cpu(), y_true=targets.cpu(), y_pred=preds.cpu())
    log.flush()


def train(model, optimizer, scheduler, dataset, args, log, use_sam=False, lambda_optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_value = lambda_optimizer.current_lambda if lambda_optimizer else args.lambda_
    ce_loss = nn.CrossEntropyLoss()

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

    for _ in range(args.epochs):
        _train_epoch(model, dataset["train"], optimizer, scheduler, criterion, device, log, use_sam)
        _evaluate(model, dataset["val"], criterion, device, log, "Validation")
        _evaluate(model, dataset["test"], criterion, device, log, "Test")

    log.attach_robust_metrics(model, dataset["test"], device, ce_loss)


def load_dataset(name, batch_size):
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

    full_train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform_test)

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
    parser.add_argument("--optimize_lambda", action="store_true")
    parser.add_argument("--lambda_", default=1.0, type=float)
    parser.add_argument("--lambda_range", default="0,1,0.2", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--loss_type", default="focal", type=str, help="Tipo di loss da combinare")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    dataset, num_classes = load_dataset(args.dataset, args.batch_size)
    model_fn = {2: WRN56_2, 4: WRN56_4, 8: WRN56_8}.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    lambda_values = [args.lambda_]
    if args.optimize_lambda:
        try:
            start, end, step = map(float, args.lambda_range.split(","))
            lambda_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]
        except ValueError:
            raise ValueError("lambda_range must be three comma-separated floats")

    print(f">>> Training with SGD\n>>> Loss configuration: {args.loss_type}")
    for lambda_val in lambda_values:
        model = model_fn(num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        log = Log(log_each=10, model_name=f"model_sgd_{lambda_val:.2f}.pth", lambda_value=lambda_val, optimize_lambda=args.optimize_lambda)

        train(model, optimizer, scheduler, dataset, args, log, use_sam=False)
      
        print(">>> Salvando il modello migliore")
        torch.save(model.state_dict(), log.best_model_path)
       
        
        log.print_best_metrics()

    print("Plotting weight analysis...")
    try:
        subprocess.run([
            "python", "weight_analysis.py",
            "--models_dir", "results/",
            "--lambda_range", args.lambda_range,
            "--depth", str(args.depth)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] plot_weight_analysis fallito: {e}")
        
        
    '''print(">>> Training with SAM , Loss configuration: {args.loss_type}")
    for lambda_val in lambda_values:
        model = model_fn(num_classes=num_classes).to(device)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer.base_optimizer, step_size=30, gamma=0.1)
        log = Log(log_each=10, model_name=f"model_sam_{lambda_val:.2f}.pth", lambda_value=lambda_val, optimize_lambda=args.optimize_lambda)

        train(model, optimizer, scheduler, dataset, args, log, use_sam=True)

        print(">>> Salvando il modello migliore")
        torch.save(model.state_dict(), log.best_model_path)
        log.print_best_metrics()'''
