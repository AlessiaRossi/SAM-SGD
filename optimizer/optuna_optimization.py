import os
import torch
from torch.optim.lr_scheduler import StepLR
import optuna
from utility.log import Log
from utility.loss import CombinedLoss, TRADESLoss
from optimizer.optimizer import create_optimizer
from train import _evaluate

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
            print(f" Nuovo miglior modello globale salvato in {best_global_model_path}")
        else:
            print("Trial peggiore del migliore globale. Nessun aggiornamento del modello.")


        return val_acc  

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_rho = study.best_params["rho"]
    best_lambda = study.best_params["lambda_"]
    print(f" Miglior combinazione trovata: rho = {best_rho:.3f}, lambda = {best_lambda:.2f}, val_acc = {-study.best_value:.4f}")
    print(f" Modello migliore salvato in: {best_global_model_path}")


    return best_rho, best_lambda