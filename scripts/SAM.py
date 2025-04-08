import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
def grid_search_sam_rho(model_fn, dataset, args, rhos=[0.01, 0.03, 0.05, 0.1]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for rho in rhos:
        print(f"\n>>> Training SAM with rho = {rho}")
        model = model_fn(num_classes=10).to(device)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=rho,
            adaptive=False,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer.base_optimizer, args.learning_rate, args.epochs)

        log_file = f"evaluation_sam_rho_{rho:.3f}.csv"
        model_name = f"model_sam_rho_{rho:.3f}.pth"
        log = Log(log_each=10, log_file=log_file, model_name=model_name, algorithm_name=f"SAM_rho_{rho:.3f}")
        train(model, optimizer, scheduler, dataset, args, log, use_sam=True)

        results.append((rho, log.best_accuracy))

    best_rho, best_acc = max(results, key=lambda x: x[1])
    print(f"\n[GRID SEARCH] Best rho: {best_rho:.3f} with Validation Accuracy: {best_acc*100:.2f}%")
    return best_rho, best_acc
