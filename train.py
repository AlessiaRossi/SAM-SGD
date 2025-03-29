import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


from model.ResNet import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
from utility.log import Log
from utility.initialize import initialize
from utility.lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from scripts.SAM import SAM

def train(model, optimizer, scheduler, dataset, args, log, use_sam=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

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

            with torch.no_grad():
                correct = torch.argmax(predictions, dim=1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr(), y_true=targets, y_pred=torch.argmax(predictions, 1))
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset["test"]))

        with torch.no_grad():
            for inputs, targets in dataset["test"]:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, dim=1) == targets
                log(model, loss.cpu(), correct.cpu(), y_true=targets, y_pred=torch.argmax(predictions, 1))

    log.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=20, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--rho", default=0.05, type=float)
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    train_dataset = CIFAR10(root="./cifar", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="./cifar", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataset = {"train": train_loader, "test": test_loader}

    resnet_versions = {
        20: ResNet20,
        32: ResNet32,
        44: ResNet44,
        56: ResNet56,
        110: ResNet110,
    }
    model_fn = resnet_versions.get(args.depth)
    if model_fn is None:
        raise ValueError(f"Unsupported depth {args.depth}")

    print(">>> Training with SGD")
    model_sgd = model_fn(num_classes=10).to(device)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_sgd = StepLR(optimizer_sgd, args.learning_rate, args.epochs)
    log_sgd = Log(log_each=10, log_file="training_sgd.csv", model_name="model_sgd.pth")
    train(model_sgd, optimizer_sgd, scheduler_sgd, dataset, args, log_sgd, use_sam=False)

    print("\n>>> Training with SAM")
    model_sam = model_fn(num_classes=10).to(device)
    base_optimizer = torch.optim.SGD
    optimizer_sam = SAM(model_sam.parameters(), base_optimizer, rho=args.rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_sam = StepLR(optimizer_sam.base_optimizer, args.learning_rate, args.epochs)
    log_sam = Log(log_each=10, log_file="training_sam.csv", model_name="model_sam.pth")
    train(model_sam, optimizer_sam, scheduler_sam, dataset, args, log_sam, use_sam=True)

    print(f"\nFinal Accuracy SGD:  {log_sgd.best_accuracy * 100:.2f}%")
    print(f"Final Accuracy SAM:  {log_sam.best_accuracy * 100:.2f}%")
