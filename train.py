# train.py (complete: dataset, SAM, Grid Search, lambda_range support for both optimizers)
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split
from model.Net import WRN56_2, WRN56_4, WRN56_8
from utility.log import Log
from utility.initialize import initialize
from utility.lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from scripts.SAM import SAM, grid_search_sam_rho
from utility.loss import CombinedLoss, FocalLoss, LogitNormLoss, TRADESLoss

def _train_epoch(model, dataloader, optimizer, scheduler, criterion, device, log, use_sam):
    model.train()
    log.train(len_dataset=len(dataloader))

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        if use_sam:
            enable_running_stats(model)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            predictions = model(inputs)
            criterion(predictions, targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

        scheduler(log.epoch)

def _evaluate(model, dataloader, criterion, device, log, split_name):
    model.eval()
    log.eval(len_dataset=len(dataloader), label=split_name)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            preds = predictions.argmax(dim=1)
            accuracy = (preds == targets).float().mean()
            batch_size = targets.size(0)
            acc_tensor = accuracy.repeat(batch_size)
            log(model, loss.cpu(), acc_tensor.cpu(), y_true=targets.cpu(), y_pred=preds.cpu())
    log.flush()

def train(model, optimizer, scheduler, dataset, args, log, use_sam=False, lambda_optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_value = lambda_optimizer.current_lambda if lambda_optimizer else args.lambda_

    # Definizione delle funzioni di perdita
    ce_loss = nn.CrossEntropyLoss()
    #focal_loss = FocalLoss(gamma=2, alpha=0.25)  # Configura Focal Loss con gamma e alpha
    #criterion = CombinedLoss(loss1=ce_loss, loss2=focal_loss, lambda_=lambda_value)
    logitnorm_loss = LogitNormLoss(device=device, t=1.0)  # Configura LogitNormLoss con temperatura t=1.0
    criterion = CombinedLoss(loss1=ce_loss, loss2=logitnorm_loss, lambda_=lambda_value)

    for _ in range(args.epochs):
        _train_epoch(model, dataset["train"], optimizer, scheduler, criterion, device, log, use_sam)
        _evaluate(model, dataset["val"], criterion, device, log, "Validation")
        _evaluate(model, dataset["test"], criterion, device, log, "Test")

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return {"train": train_loader, "val": val_loader, "test": test_loader}, num_classes

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
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    dataset, num_classes = load_dataset(args.dataset, args.batch_size)
    wresnet_versions = {2: WRN56_2, 4: WRN56_4, 8: WRN56_8}
    model_fn = wresnet_versions.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    lambda_values = [args.lambda_]
    if args.optimize_lambda:
        try:
            lambda_start, lambda_end, lambda_step = map(float, args.lambda_range.split(","))
            step_count = int((lambda_end - lambda_start) / lambda_step) + 1
            lambda_values = [round(lambda_start + i * lambda_step, 2) for i in range(step_count)]
        except ValueError:
            raise ValueError("lambda_range must be three comma-separated floats: start,end,step")

    '''print(">>> Training with SGD")
    for lambda_val in lambda_values:
        model_sgd = model_fn(num_classes=num_classes).to(device)
        optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_sgd = StepLR(optimizer_sgd, args.learning_rate, args.epochs)
        log_sgd = Log(log_each=10, log_file=f"evaluation_sgd_lambda_{lambda_val:.2f}.csv", model_name=f"model_sgd_{lambda_val:.2f}.pth", lambda_value=lambda_val, optimize_lambda=args.optimize_lambda)
        train(model_sgd, optimizer_sgd, scheduler_sgd, dataset, args, log_sgd, use_sam=False, lambda_optimizer=None)
        log_sgd.save_best_metrics()
        log_sgd.print_best_metrics()'''

    print(">>> Training with SAM")
    for lambda_val in lambda_values:
        model_sam = model_fn(num_classes=num_classes).to(device)
        base_optimizer = torch.optim.SGD
        optimizer_sam = SAM(model_sam.parameters(), base_optimizer, rho=args.rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_sam = StepLR(optimizer_sam.base_optimizer, args.learning_rate, args.epochs)
        log_sam = Log(log_each=10, log_file=f"evaluation_sam_lambda_{lambda_val:.2f}.csv", model_name=f"model_sam_{lambda_val:.2f}.pth", lambda_value=lambda_val, optimize_lambda=args.optimize_lambda)
        train(model_sam, optimizer_sam, scheduler_sam, dataset, args, log_sam, use_sam=True, lambda_optimizer=None)
        log_sam.save_best_metrics()
        log_sam.print_best_metrics()

    print(">>> Grid Search for SAM rho")
    best_rho, best_acc = grid_search_sam_rho(model_fn, dataset, args, rhos=[0.01, 0.03, 0.05, 0.1], train_fn=train)
    print(f"\nBest rho: {best_rho} with Validation Accuracy: {best_acc * 100:.2f}%")
