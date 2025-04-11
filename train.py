import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from model.Net import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, WRN56_2, WRN56_4, WRN56_8
from utility.log import Log
from utility.initialize import initialize
from utility.lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from scripts.SAM import SAM, grid_search_sam_rho
from utility.loss import CE_FL_Loss, LambdaOptimizer


def train(model, optimizer, scheduler, dataset, args, log, use_sam=False, lambda_optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CE_FL_Loss(lambda_=lambda_optimizer.current_lambda if lambda_optimizer else args.lambda_)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset["train"]))

        for inputs, targets in dataset["train"]:
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

            scheduler(epoch)

        # Validazione
        model.eval()
        log.eval(len_dataset=len(dataset["val"]), label="Validation")
        with torch.no_grad():
            for inputs, targets in dataset["val"]:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                preds = torch.argmax(predictions, dim=1)
                correct = (preds == targets).float()
                batch_size = targets.size(0)
                acc_tensor = torch.full((batch_size,), correct.sum().item() / batch_size)
                log(model, loss.cpu(), acc_tensor.cpu(), y_true=targets.cpu(), y_pred=preds.cpu())

        log.flush()

        # Fase di test
        model.eval()
        log.eval(len_dataset=len(dataset["test"]), label="Test")
        with torch.no_grad():
            for inputs, targets in dataset["test"]:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                preds = torch.argmax(predictions, dim=1)
                correct = (preds == targets).float()
                batch_size = targets.size(0)
                acc_tensor = torch.full((batch_size,), correct.sum().item() / batch_size)
                log(model, loss.cpu(), acc_tensor.cpu(), y_true=targets.cpu(), y_pred=preds.cpu())

        log.flush()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=2, type=int)  # 20 default per resnet, 2 default per wresnet
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--rho", default=0.05, type=float)
    parser.add_argument("--optimize_lambda", action="store_true", help="Optimize lambda using Optuna")
    parser.add_argument("--lambda_", default=1.0, type=float, help="Default lambda value for loss function")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

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

    full_train_dataset = CIFAR10(root="./cifar", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="./cifar", train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(full_train_dataset))  # 45k
    val_size = len(full_train_dataset) - train_size  # 5k
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataset = {"train": train_loader, "val": val_loader, "test": test_loader}

    '''resnet_versions = {
        20: ResNet20,
        32: ResNet32,
        44: ResNet44,
        56: ResNet56,
        110: ResNet110,
    }
    model_fn = resnet_versions.get(args.depth)'''

    wresnet_versions = {
        2: WRN56_2,
        4: WRN56_4,
        8: WRN56_8
    }

    model_fn = wresnet_versions.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    print(">>> Training with SGD")
    model_sgd = model_fn(num_classes=10).to(device)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_sgd = StepLR(optimizer_sgd, args.learning_rate, args.epochs)
    log_sgd = Log(log_each=10, log_file="evaluation_sgd.csv", model_name="model_sgd.pth", lambda_value=args.lambda_,  # Passa il valore aggiornato di lambda_
    optimize_lambda=args.optimize_lambda)

    if args.optimize_lambda:
        # Inizializza l'ottimizzatore di lambda
        lambda_optimizer = LambdaOptimizer(
            train_fn=train,
            dataset=dataset,
            model=model_sgd,
            optimizer=optimizer_sgd,
            scheduler=scheduler_sgd,
            args=args,
            step=0.2
        )
        lambda_optimizer.run()  # Esegui il training per ogni valore di lambda
    else:
        print(f"Using default lambda = {args.lambda_}")

        # Esegui il training con il valore di lambda di default
        train(model_sgd, optimizer_sgd, scheduler_sgd, dataset, args, log_sgd, use_sam=False, lambda_optimizer=None)
        log_sgd.save_best_metrics()
        log_sgd.print_best_metrics()

    '''print("\n>>> Training with SAM")
    model_sam = model_fn(num_classes=10).to(device)
    base_optimizer = torch.optim.SGD
    optimizer_sam = SAM(model_sam.parameters(), base_optimizer, rho=args.rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_sam = StepLR(optimizer_sam.base_optimizer, args.learning_rate, args.epochs)
    log_sam = Log(log_each=10, log_file="evaluation_sam.csv", model_name="model_sam.pth",  lambda_value=args.lambda_,  # Passa il valore aggiornato di lambda_
    optimize_lambda=args.optimize_lambda)
    train(model_sam, optimizer_sam, scheduler_sam, dataset, args, log_sam, use_sam=True, lambda_=args.lambda_, loss_type="cross_entropy+focal_loss")
    log_sam.save_best_metrics()
    log_sam.print_best_metrics()'''

    #print("\n>>> Grid Search for SAM rho")
    #best_rho, best_acc = grid_search_sam_rho(model_fn, dataset, args, rhos=[0.01, 0.03, 0.05, 0.1])
    #print(f"\nBest rho: {best_rho} with Validation Accuracy: {best_acc * 100:.2f}%")

    
    print(f"\nFinal Accuracy SGD:  {log_sgd.best_metrics['test_accuracy'] * 100:.2f}%")
    #print(f"Final Accuracy SAM:  {log_sam.best_metrics['test_accuracy'] * 100:.2f}%")